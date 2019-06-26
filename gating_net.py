import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.distributions import normal


class GatingNet(nn.Module):
    def __init__(self, cuda=None, tasks=None, blocks=3):
        super(GatingNet, self).__init__()
        self.active_tasks = 1
        self.cuda = cuda
        self.blocks = blocks
        self.normal = normal.Normal(0.0, 0.0001)
        self.tasks = tasks
        self.gate_log = torch.zeros([len(self.tasks), self.blocks])
        self.gate_count_log = torch.zeros([len(self.tasks)])

        if cuda:
            self.gate_log = self.gate_log.cuda(cuda)
            self.gate_count_log = self.gate_count_log.cuda(cuda)

    def setup(self, layered=False, optimizer=None, criterion=None):

        if layered == False:
            self.g_logits = nn.Parameter(torch.zeros([len(self.tasks), self.blocks]))
            self.q_logits = nn.Parameter(torch.zeros([len(self.tasks), 2, self.blocks]))
            self.mix_logits = nn.Parameter(torch.zeros([len(self.tasks), 2 ** self.blocks]))

        self.config = torch.zeros(self.blocks, 1, 1, self.blocks, 1)
        for c in range(self.blocks):
            for b in range(self.blocks):
                if c == b:
                    self.config[c, 0, 0, b, 0] = 1
        self.mix_config = torch.zeros(2 ** self.blocks, 1, 1, self.blocks, 1)
        for c in range(2 ** self.blocks):
            for b in range(self.blocks):
                if (c // (2**b)) % 2 == 1:
                    self.mix_config[c, 0, 0, b, 0] = 1

        self.mix_map = torch.zeros(2 ** self.blocks, self.blocks)
        for c in range(2 ** self.blocks):
            for b in range(self.blocks):
                if (c // (2 ** b)) % 2 == 1:
                    self.mix_map[c, b] = 1


        if self.cuda:
            self.config = self.config.cuda(self.cuda)
            self.mix_config = self.mix_config.cuda(self.cuda)

        if optimizer is None:
            self.optimizer = optim.Adam(self.parameters())
        if criterion is None:
            self.criterion = nn.CrossEntropyLoss(reduction='none')
            if self.cuda:
                self.criterion.cuda(self.cuda)

    def get_losses(self, inputs, task_labels, tasks=None, naive_sample=False, sample=False, individual=False, mixture=False, VI=False):
        if naive_sample:  # sample
            outputs, probs = self(inputs, tasks=tasks, naive_sample=naive_sample, sample=sample, individual=individual,
                                  mixture=mixture, VI=VI)
            losses = [torch.stack(
                [self.criterion(outputs[t][c], task_labels[t]) * probs[t][c] for c in range(self.blocks)]).sum(
                dim=0)
            for t in range(len(outputs))]
        elif sample:
            outputs, log_probs = self(inputs, tasks=tasks, naive_sample=naive_sample, sample=sample, individual=individual,
                                  mixture=mixture, VI=VI)
            losses = [torch.stack(
                [self.criterion(outputs[t][c], task_labels[t]) + log_probs[t][c] for c in range(self.blocks)]).logsumexp(
                dim=0)
            for t in range(len(outputs))]
        elif VI:
            outputs, log_p, log_q, q_probs = self(inputs, tasks=tasks, naive_sample=naive_sample, sample=sample, individual=individual, mixture=mixture,
                                                          VI=VI, labels=task_labels)
            losses = []
            for t in range(len(outputs)):
                batch_s = task_labels[t].shape[0]

                log_qs = torch.gather(log_q[t][None].repeat(batch_s, 1, 1), 1,
                                      task_labels[t].reshape(batch_s, 1, 1).repeat(1, 1, self.blocks))
                qs = torch.gather(q_probs[t][None].repeat(batch_s, 1, 1), 1,
                                  task_labels[t].reshape(batch_s, 1, 1).repeat(1, 1, self.blocks))

                log_qs = log_qs.reshape(batch_s, -1)
                qs = qs.reshape(batch_s, -1)

                losses.append(-(torch.stack(
                    [(-self.criterion(outputs[t][c], task_labels[t]) + log_p[t, c] - log_qs[:, c]) * qs[:, c] for c in
                     range(self.blocks)]).sum()))
        else:  # blend/individual
            outputs = self(inputs, tasks=tasks, naive_sample=naive_sample, sample=sample, individual=individual, mixture=mixture, VI=VI)
            losses = [self.criterion(out, label).mean() for out, label in zip(outputs, task_labels)]
        return losses


    def train_epoch(self, sample=False, naive_sample=False, individual=False, mixture=False, VI=False, log_interval=1000):
        print("Starting Training")
        if self.log:
            print("Starting Training", file=open(self.log, "a"))
        running_loss_total = 0.0
        for i, data in enumerate(self.trainloader, 0):
            # get the inputs
            if len(data) == 2:
                inputs, labels = data
                tasks = None
            else:
                inputs, tasks, labels = data

            if self.cuda is not None:
                inputs = inputs.cuda(self.cuda)
                labels = labels.cuda(self.cuda)
                if tasks is not None:
                    tasks = tasks.cuda(self.cuda)

            task_labels = [torch.tensor([self.tasks[t].label_to_task(l) for l in labels], dtype=torch.long) for t in range(self.active_tasks)]

            if self.cuda is not None:
                task_labels = [t.cuda(self.cuda) for t in task_labels]

            # zero the parameter gradients
            self.optimizer.zero_grad()

            # get non-gate parameter loss
            losses = self.get_losses(inputs, task_labels, tasks=tasks, sample=sample, naive_sample=naive_sample, individual=individual, mixture=mixture, VI=VI)

            scales = [t.scale for t in self.tasks]
            net_loss = sum([losses[i] * scales[i] for i in range(len(self.tasks))]) / len(losses)

            #get gate loss
            loss = net_loss
            loss.backward()
            self.optimizer.step()

            # print statistics
            with torch.no_grad():
                running_loss_total += loss.item()
                if i % log_interval == log_interval-1:
                    if self.log:
                        print('[%5d] loss: %.3f' % (i + 1, running_loss_total / log_interval), file=open(self.log, "a"))
                    else:
                        print('[%5d] loss: %.3f' % (i + 1, running_loss_total / log_interval))
                    running_loss_total = 0.0

                    for name, param in self.named_parameters():
                        if param.requires_grad:
                            if not individual and not mixture:
                                if name in ["g_logits", "g_logits1", "g_logits2", "g_logits3"]:
                                    if self.log:
                                        print(name, file=open(self.log, "a"))
                                        for i in range(len(param.data)):
                                            print(list(nn.functional.softmax(param.data[i], dim=0)), file=open(self.log, "a"))
                                        #for i in range(len(self.tasks)):
                                        #    print("grad:", self.gate_log[i] / self.gate_count_log[i], file=open(self.log, "a"))
                                    print(name)
                                    for i in range(len(param.data)):
                                        print(list(nn.functional.softmax(param.data[i], dim=0)))
                                    #for i in range(len(self.tasks)):
                                    #    print("grad:", self.gate_log[i] / self.gate_count_log[i])

                                    self.gate_log.zero_()
                                    self.gate_count_log.zero_()

                            if mixture:
                                if name == "mix_logits":
                                    if self.log:
                                        print(name, param.data, file=open(self.log, "a"))
                                        for i in range(len(self.tasks)):
                                            print("mix:", nn.functional.softmax(param.data[i], dim=0), file=open(self.log, "a"))

                                    print(name, param.data)
                                    for i in range(len(self.tasks)):
                                        print("mix:", nn.functional.softmax(param.data[i], dim=0))
                            if VI:
                                if name in ["q_logits", "q_logits1", "q_logits2", "q_logits3"]:
                                    if self.log:
                                        print(name, file=open(self.log, "a"))
                                        for i in range(len(param.data)):
                                            print(list(nn.functional.softmax(param.data[i], dim=1)),
                                                  file=open(self.log, "a"))
                                        # for i in range(len(self.tasks)):
                                        #    print("grad:", self.gate_log[i] / self.gate_count_log[i], file=open(self.log, "a"))
                                    print(name)
                                    for i in range(len(param.data)):
                                        print(list(nn.functional.softmax(param.data[i], dim=1)))

        print('Finished Training')
        if self.log:
            print("Finished Training", file=open(self.log, "a"))

    def test_overall(self, sample=False, individual=False, naive_sample=False, mixture=False, VI=False):
        self.eval()
        correct_arr = np.array([0 for _ in range(len(self.tasks))])
        total_arr = np.array([0 for _ in range(len(self.tasks))])

        total_losses = torch.zeros(len(self.tasks))

        if self.cuda is not None:
            total_losses = total_losses.cuda(self.cuda)

        i = 0

        with torch.no_grad():
            for data in self.testloader:

                if len(data) == 2:
                    inputs, labels = data
                    tasks = None
                else:
                    inputs, tasks, labels = data

                if self.cuda is not None:
                    inputs = inputs.cuda(self.cuda)
                    labels = labels.cuda(self.cuda)

                task_labels = [torch.tensor([self.tasks[t].label_to_task(l) for l in labels], dtype=torch.long) for t in range(self.active_tasks)]
                if self.cuda is not None:
                    task_labels = [t.cuda(self.cuda) for t in task_labels]

                outputs = self.forward(inputs, test=True, sample=sample, individual=individual, mixture=mixture, VI=VI)
                #losses = self.get_losses(inputs, task_labels, tasks=tasks, naive_sample=naive_sample, sample=sample, individual=individual, mixture=mixture,
                #                         VI=VI)

                #total_losses += losses
                #i += 1
                predicted = [torch.max(nn.functional.softmax(output, dim=1).data, 1)[1] for output in outputs]

                if tasks is not None:
                    for i in range(len(tasks)):
                        total_arr[tasks[i]] += 1
                        correct_arr[tasks[i]] += float(predicted[tasks[i]][i] == task_labels[tasks[i]][i])
                else:
                    for t in range(len(self.tasks)):
                        total_arr[t] += len(predicted[t])
                        correct_arr[t] += float(torch.sum(predicted[t] == task_labels[t]))

        #total_losses = total_losses / i

        if self.log:
            for i in range(len(self.tasks)):
                print('task_' + str(i) + ' accuracy of the network on test images: %.2f %%' % (
                        100 * correct_arr[i] / total_arr[i]), file=open(self.log, "a"))
                #print('task_' + str(i) + ' loss of the network on test images: %f' % (
                #        total_losses[i]), file=open(self.log, "a"))
            print('overall accuracy of the network on the 10000 test images: %.2f %%' % (
                    100 * sum(correct_arr) / sum(total_arr)), file=open(self.log, "a"))

        for i in range(len(self.tasks)):
            print('task_' + str(i) + ' accuracy of the network on the 10000 test images: %.2f %%' % (
                    100 * correct_arr[i] / total_arr[i]))
            #print('task_' + str(i) + ' loss of the network on test images: %f' % (
            #        total_losses[i]))
        print('overall accuracy of the network on the 10000 test images: %.2f %%' % (
                100 * sum(correct_arr) / sum(total_arr)))

        self.train()

    def log_gate(self, v, i):
        self.gate_log[i] += v

    def add_normal_noise(self, v):
        return v + self.normal.sample()

    def log_assign(self, i):
        return lambda grad: self.log_gate(grad, i)

    def forward(self, img, tasks=None, test=False, sample=False, naive_sample=False, individual=False, mixture=False, VI=False, labels=None):

        blocks = [self.convs[i](img) for i in range(self.blocks)]
        stack = torch.stack([t.view(-1, self.fc_out) for t in blocks], dim=1) #batch_size x blocks x vals

        """
        # Sample and return g1, g2, g4

        g1 = torch.distributions.OneHotCategorical(logits=self.g_logits[0]).sample(
            [img.shape[0]]) # batch_size x num_blocks
        g2 = torch.distributions.OneHotCategorical(logits=self.g_logits[1]).sample(
            [img.shape[0]])  # batch_size x num_blocks
        g4 = torch.distributions.OneHotCategorical(logits=self.g_logits[2]).sample(
            [img.shape[0]])  # batch_size x num_blocks
        

        prob_g1 = (nn.functional.softmax(self.g_logits[0], dim=0) * g1).sum(dim=1)
        prob_g2 = (nn.functional.softmax(self.g_logits[1], dim=0) * g2).sum(dim=1)
        prob_g3 = (nn.functional.softmax(self.g_logits[2], dim=0) * g2).sum(dim=1)
        """

        if naive_sample:
            stack = stack[None].repeat(len(self.tasks), 1, 1, 1)
            probs = nn.functional.softmax(self.g_logits, dim=1)  # numtasks x blocks

            # config: 3 x 3 x 1 x 3 x 1
            # stack:      3 x 1 x 1 x 1
            fc_ins = (stack * self.config).sum(dim=3)
            fc_ins = fc_ins.permute(1, 0, 2, 3)

            # fc_outs = [self.fcs[i](stack.view(-1, self.fc_out)).view(32, self.blocks, 2)]

        elif sample:
            #if self.training:
                #print("sample training")
            stack = stack[None].repeat(len(self.tasks), 1, 1, 1)
            log_probs = nn.functional.log_softmax(self.g_logits, dim=1) #numtasks x blocks

            #config: 3 x 3 x 1 x 3 x 1
            #stack:      3 x 1 x 1 x 1
            fc_ins = (stack * self.config).sum(dim=3)
            fc_ins = fc_ins.permute(1, 0, 2, 3)

            #fc_outs = [self.fcs[i](stack.view(-1, self.fc_out)).view(32, self.blocks, 2)]
            """
            else:
                #print("sample testing")
                
                sample_gs = [torch.distributions.OneHotCategorical(logits=self.g_logits[i]).sample(
                    [img.shape[0]]) for i in range(len(self.tasks))]  # tasks x blocks x batch_size
                fc_ins = [(stack * g.reshape(g.shape[0], g.shape[1], 1)).sum(dim=1) for g in sample_gs]
                fc_outs = [self.fcs[i](fc_ins[i]) for i in range(len(self.tasks))]
                
            """
        elif VI:
            if self.training:
                stack = stack[None].repeat(len(self.tasks), 1, 1, 1)
                log_p = nn.functional.log_softmax(self.g_logits, dim=1)
                log_q = nn.functional.log_softmax(self.q_logits, dim=2)
                q_probs = nn.functional.softmax(self.q_logits, dim=2)

                # 3 x 6 x 32 x 3 x 3200
                fc_ins = (stack * self.config).sum(dim=3)  # tasks x stack
                fc_ins = fc_ins.permute(1, 0, 2, 3)
                # 6 x 3 x 32 x 3200

                # fc_outs = [self.fcs[i](stack.view(-1, self.fc_out)).view(32, self.blocks, 2)]
            else:
                """
                sample_gs = [torch.distributions.OneHotCategorical(logits=self.g_logits[i]).sample(
                    [img.shape[0]]) for i in range(len(self.tasks))]  # tasks x blocks x batch_size
                fc_ins = [(stack * g.reshape(g.shape[0], g.shape[1], 1)).sum(dim=1) for g in sample_gs]
                fc_outs = [self.fcs[i](fc_ins[i]) for i in range(len(self.tasks))]
                """
                stack = stack[None].repeat(len(self.tasks), 1, 1, 1)
                log_p = nn.functional.log_softmax(self.g_logits, dim=1)
                log_q = nn.functional.log_softmax(self.q_logits, dim=2)
                q_probs = nn.functional.softmax(self.q_logits, dim=2)

                # 3 x 6 x 32 x 3 x 3200
                fc_ins = (stack * self.config).sum(dim=3)  # tasks x stack
                fc_ins = fc_ins.permute(1, 0, 2, 3)
                # 6 x 3 x 32 x 3200

                # fc_outs = [self.fcs[i](stack.view(-1, self.fc_out)).view(32, self.blocks, 2)]
        elif individual:
            fc_ins = [blocks[i].view(-1, self.fc_out) for i in range(len(self.tasks))]
        elif mixture:
            if self.training:
                log_probs = nn.functional.log_softmax(self.mix_logits, dim=1)
                #gates = nn.functional.softmax(self.g_logits, dim=1)
                stack = stack.reshape(1, stack.shape[0], stack.shape[1], stack.shape[2]).repeat(len(self.tasks), 1, 1, 1)
                #stack = stack * gates.view(len(self.tasks), 1, self.blocks, 1)
                fc_ins = (stack * self.mix_config).sum(dim=3)  # tasks x stack
                fc_ins = fc_ins.permute(1, 0, 2, 3) #task x config x batch_size x block_size
            else:
                """
                gates = [nn.functional.softmax(self.g_logits[i], dim=0) for i in range(len(self.tasks))]
                sample_gs = [torch.mm(torch.distributions.OneHotCategorical(logits=self.mix_logits[i]).sample(
                    [img.shape[0]]), self.mix_map) for i in range(len(self.tasks))]  # tasks x 2**blocks x batch_size

                fc_ins = [(stack * sample_gs[i].reshape(sample_gs[i].shape[0], sample_gs[i].shape[1], 1) *
                           gates[i].view(-1, 1)).sum(dim=1) for i in range(len(sample_gs))]
                fc_outs = [self.fcs[i](fc_ins[i]) for i in range(len(self.tasks))]
                """
                log_probs = nn.functional.log_softmax(self.mix_logits, dim=1)
                # gates = nn.functional.softmax(self.g_logits, dim=1)
                stack = stack.reshape(1, stack.shape[0], stack.shape[1], stack.shape[2]).repeat(len(self.tasks), 1, 1,
                                                                                                1)
                # stack = stack * gates.view(len(self.tasks), 1, self.blocks, 1)
                fc_ins = (stack * self.mix_config).sum(dim=3)  # tasks x stack
                fc_ins = fc_ins.permute(1, 0, 2, 3)  # task x config x batch_size x block_size
        else: #blending
            gates = [nn.functional.softmax(self.g_logits[i], dim=0) for i in range(len(self.tasks))]
            fc_ins = [(stack * g.view(-1, 1)).sum(dim=1) for g in gates]

        fc_outs = torch.stack([self.fcs[i](fc_ins[i]) for i in range(len(self.tasks))])

        if naive_sample:
            return fc_outs, probs
        elif (sample or mixture):
            return fc_outs, log_probs
        elif VI:
            return fc_outs, log_p, log_q, q_probs
        else:
            return fc_outs


class LargeNet(GatingNet):
    def __init__(self, tasks, trainloader, testloader, fc_out, in_channels, log=None, cuda=None, conv_noise=False, gate_noise=False, blocks=3):
        super(LargeNet, self).__init__(cuda=cuda, tasks=tasks, blocks=blocks)
        self.trainloader = trainloader
        self.testloader = testloader
        self.log = log
        self.fc_out = fc_out
        self.conv_noise = conv_noise
        self.gate_noise = gate_noise

        self.convs = nn.ModuleList([nn.Sequential(
            nn.Conv2d(in_channels, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(128, affine=False)
        ) for _ in range(self.blocks)])

        self.fcs = nn.ModuleList([nn.Sequential(
            nn.Linear(self.fc_out, 16),
            nn.Linear(16, tasks[i].size)
        ) for i in range(len(self.tasks))])

        self.setup()


class LargerNet(GatingNet):
    def __init__(self, tasks, trainloader, testloader, fc_out, in_channels, log=None, cuda=None, conv_noise=False, gate_noise=False, blocks=3):
        super(LargerNet, self).__init__(cuda=cuda, tasks=tasks, blocks=blocks)
        self.trainloader = trainloader
        self.testloader = testloader
        self.log = log
        self.fc_out = fc_out
        self.conv_noise = conv_noise
        self.gate_noise = gate_noise

        self.convs = nn.ModuleList([nn.Sequential(
            nn.Conv2d(in_channels, 256, 3),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(512, affine=False)
        ) for _ in range(self.blocks)])

        self.fcs = nn.ModuleList([nn.Sequential(
            nn.Linear(self.fc_out, 16),
            nn.Linear(16, tasks[i].size)
        ) for i in range(len(self.tasks))])

        self.setup()


class AsymmNet(GatingNet):
    def __init__(self, tasks, trainloader, testloader, fc_out, in_channels, log=None, cuda=None, conv_noise=False, gate_noise=False, blocks=3):
        super(AsymmNet, self).__init__(cuda=cuda, tasks=tasks, blocks=blocks)
        self.trainloader = trainloader
        self.testloader = testloader
        self.log = log
        self.fc_out = fc_out
        self.conv_noise = conv_noise
        self.gate_noise = gate_noise

        self.convs = nn.ModuleList([nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, dilation=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, dilation=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(128, affine=False)
        ), nn.Sequential(
            nn.Conv2d(in_channels, 64, 5),
            nn.ReLU(),
            nn.Conv2d(64, 64, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(128, affine=False)
        ), nn.Sequential(
            nn.Conv2d(in_channels, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            nn.Conv2d(128, 128, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(128, affine=False)
        )])

        self.fcs = nn.ModuleList([nn.Sequential(
            nn.Linear(self.fc_out, 16),
            nn.Linear(16, tasks[i].size)
        ) for i in range(len(self.tasks))])

        self.setup()


class SimpleConvNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel):
        nn.Module.__init__(self)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        y = self.conv(x)
        y = self.relu(y)
        y = self.maxpool(y)
        return y

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size()[0], -1)

class RouteNet(GatingNet):
    def __init__(self, tasks, trainloader, testloader, fc_out, in_channels, log=None, cuda=None, conv_noise=False, gate_noise=False, blocks=3):
        self.active_tasks = 1
        super(RouteNet, self).__init__(cuda=cuda, tasks=tasks, blocks=blocks)
        self.trainloader = trainloader
        self.testloader = testloader
        self.log = log
        self.fc_out = fc_out
        self.conv_noise = conv_noise
        self.gate_noise = gate_noise

        self.convolutions = nn.Sequential(
            SimpleConvNetBlock(in_channels, 32, 3),
            SimpleConvNetBlock(32, 32, 3),
            SimpleConvNetBlock(32, 32, 3),
            SimpleConvNetBlock(32, 32, 3),
            nn.BatchNorm2d(32),
            Flatten()
        )

        self.fc_layer1 = nn.ModuleList([nn.Sequential(
            nn.Linear(self.fc_out, 64),
            nn.ReLU()
        ) for _ in range(self.blocks)])

        self.fc_layer2 = nn.ModuleList([nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU()
        ) for _ in range(self.blocks)])

        self.fc_layer3 = nn.ModuleList([nn.Sequential(
            nn.Linear(64, tasks[i].size)
        ) for i in range(len(self.tasks))])

        self.g_logits1 = nn.Parameter(torch.zeros([self.blocks, self.blocks]))
        self.q_logits1 = nn.Parameter(torch.zeros([self.blocks, 2, self.blocks]))

        self.g_logits2 = nn.Parameter(torch.zeros([len(self.tasks), self.blocks]))
        self.q_logits2 = nn.Parameter(torch.zeros([len(self.tasks), 2, self.blocks]))

        self.g_logits3 = nn.Parameter(torch.zeros([len(self.tasks), len(self.tasks)]))
        self.q_logits3 = nn.Parameter(torch.zeros([len(self.tasks), 2, len(self.tasks)]))

        self.setup(layered=True)

    def get_losses(self, inputs, task_labels, tasks=None, naive_sample=False, sample=False, individual=False, mixture=False, VI=False):
        outputs = self(inputs, tasks=tasks, naive_sample=naive_sample, sample=sample, individual=individual,
                       mixture=mixture, VI=VI)

        losses = torch.zeros(len(self.tasks))

        if self.cuda is not None:
            losses = losses.cuda(self.cuda)

        if naive_sample:

            soft_matrix = (F.softmax(self.g_logits3, dim=1)[:, :, None] \
                           * F.softmax(self.g_logits2, dim=1))[:, :, :, None] * F.softmax(self.g_logits1, dim=1)

            # 20 x 20 x 5 x 5
            for t in range(len(tasks)): # for each element in our batch:
                task = tasks[t]
                input = outputs[:, :, :, t][None].permute(0, 4, 1, 2, 3)
                label = task_labels[0][t].reshape(1).expand(1, len(self.tasks), self.blocks, self.blocks)


                losses[task] += (self.criterion(input, label) * soft_matrix[task][None]).sum()

        elif sample:

            soft_log_matrix = (F.log_softmax(self.g_logits3, dim=1)[:, :, None] \
                           + F.log_softmax(self.g_logits2, dim=1))[:, :, :, None] + F.log_softmax(self.g_logits1, dim=1)

            # 20 x 20 x 5 x 5
            for t in range(len(tasks)):  # for each element in our batch:
                task = tasks[t]
                input = outputs[:, :, :, t][None].permute(0, 4, 1, 2, 3)
                label = task_labels[0][t].reshape(1).expand(1, len(self.tasks), self.blocks, self.blocks)

                losses[task] += (self.criterion(input, label) + soft_log_matrix[task][None]).reshape(-1).logsumexp(dim=0)

        elif VI:

            log_p_matrix = (F.log_softmax(self.g_logits3, dim=1)[:, :, None] \
                        + F.log_softmax(self.g_logits2, dim=1))[:, :, :, None] + F.log_softmax(self.g_logits1, dim=1)

            # 20 x 20 x 5 x 5
            for t in range(len(tasks)):  # for each element in our batch:
                task = tasks[t]

                log_q_matrix = (F.log_softmax(self.q_logits3[:,task_labels[0][t],:], dim=1)[:, :, None] \
                                + F.log_softmax(self.q_logits2[:,task_labels[0][t],:], dim=1))[:, :, :, None] + F.log_softmax(self.q_logits1[:,task_labels[0][t],:],
                                                                                                       dim=1)

                q_matrix = (F.softmax(self.q_logits3[:,task_labels[0][t],:], dim=1)[:, :, None] \
                            * F.softmax(self.q_logits2[:,task_labels[0][t],:], dim=1))[:, :, :, None] * F.softmax(self.q_logits1[:,task_labels[0][t],:], dim=1)

                input = outputs[:, :, :, t][None].permute(0, 4, 1, 2, 3)
                label = task_labels[0][t].reshape(1).expand(1, len(self.tasks), self.blocks, self.blocks)

                losses[task] += ((self.criterion(input, label) - log_p_matrix[task][None] + log_q_matrix[task][None]) * q_matrix[task][None]).sum()

        else:  # blend/individual
            for t in range(len(tasks)):  # for each element in our batch:
                task = tasks[t]
                losses[task] += self.criterion(outputs[task, t].reshape(1, -1), task_labels[0][t].reshape(1)).sum()

        return losses

    def forward(self, img, tasks=None, test=False, sample=False, naive_sample=False, mixture=False, individual=False,
                VI=False, labels=None):

        layers = [self.fc_layer2, self.fc_layer3]
        g_logits = [self.g_logits1, self.g_logits2, self.g_logits3]
        blocks = self.convolutions(img)
        blocks = torch.stack([self.fc_layer1[i](blocks) for i in range(len(self.fc_layer1))], dim=0)

        for l in range(len(layers)+1):

            if not (sample or naive_sample or mixture or VI or individual):
                stack = torch.stack([t for t in blocks], dim=1)
                gates = [nn.functional.softmax(g_logits[l][i], dim=0) for i in range(len(g_logits[l]))]
                blocks = torch.stack([(stack * g.view(-1, 1)).sum(dim=1) for g in gates])

            if l < len(layers): #apply FC if not layer
                #print("applying layer, ", l)
                layer = layers[l]
                if individual or not (sample or naive_sample or VI):
                    #print(blocks.shape)
                    blocks = torch.stack([layer[i](blocks[i]) for i in range(len(layer))], dim=0)
                else:
                    #print(blocks.shape)
                    blocks = torch.stack([layer[i](blocks) for i in range(len(layer))], dim=0)

            #print("blocks:", blocks.shape)

        fc_outs = blocks


        return fc_outs


    def test_overall(self, sample=False, individual=False, naive_sample=False, mixture=False, VI=False):
        self.eval()
        correct_arr = [0 for _ in range(len(self.tasks))]
        total_arr = [0 for _ in range(len(self.tasks))]

        correct_arr = np.array(correct_arr).astype(float)
        total_arr = np.array(total_arr).astype(float)

        total_losses = torch.zeros(len(self.tasks))

        if self.cuda is not None:
            total_losses = total_losses.cuda(self.cuda)

        i = 0

        with torch.no_grad():
            for data in self.testloader:

                if len(data) == 2:
                    inputs, labels = data
                    tasks = None
                else:
                    inputs, tasks, labels = data

                if self.cuda is not None:
                    inputs = inputs.cuda(self.cuda)
                    labels = labels.cuda(self.cuda)

                task_labels = [torch.tensor([self.tasks[t].label_to_task(l) for l in labels], dtype=torch.long) for t in range(self.active_tasks)]

                if self.cuda is not None:
                    task_labels = [t.cuda(self.cuda) for t in task_labels]

                outputs = self.forward(inputs, test=True, sample=sample, naive_sample=naive_sample, individual=individual, mixture=mixture, VI=VI)

                if naive_sample or sample or VI:
                    for t in range(len(tasks)):
                        task = tasks[t]
                        dim1 = torch.distributions.Categorical(logits=self.g_logits3[task]).sample()
                        dim2 = torch.distributions.Categorical(logits=self.g_logits2[dim1]).sample()
                        dim3 = torch.distributions.Categorical(logits=self.g_logits1[dim2]).sample()

                        out = outputs[dim1, dim2, dim3, t]
                        predicted = torch.argmax(nn.functional.softmax(out, dim=0))
                        total_arr[task] += 1
                        correct_arr[task] += float(predicted == task_labels[0][t])


                else:  # blend/individual
                    for t in range(len(tasks)):
                        task = tasks[t]
                        predicted = torch.argmax(nn.functional.softmax(outputs[task][t], dim=0))
                        total_arr[task] += 1
                        correct_arr[task] += float(predicted == task_labels[0][t])


        #total_losses = total_losses / i

        if self.log:
            for i in range(len(self.tasks)):
                print('task_' + str(i) + ' accuracy of the network on test images: %.2f %%' % (
                        100 * correct_arr[i] / total_arr[i]), file=open(self.log, "a"))
                #print('task_' + str(i) + ' loss of the network on test images: %f' % (
                #        total_losses[i]), file=open(self.log, "a"))

            print('overall accuracy of the network on the 10000 test images: %.2f %%' % (
                    100 * sum(correct_arr) / sum(total_arr)), file=open(self.log, "a"))

        for i in range(len(self.tasks)):
            print('task_' + str(i) + ' accuracy of the network on the 10000 test images: %.2f %%' % (
                    100 * correct_arr[i] / total_arr[i]))
            #print('task_' + str(i) + ' loss of the network on test images: %f' % (
            #        total_losses[i]))

        print('overall accuracy of the network on the 10000 test images: %.2f %%' % (
                    100 * sum(correct_arr) / sum(total_arr)))

        self.train()


    def train_epoch(self, sample=False, naive_sample=False, individual=False, mixture=False, VI=False, log_interval=1000):
        print("Starting Training")
        if self.log:
            print("Starting Training", file=open(self.log, "a"))
        running_loss_total = 0.0

        for i, data in enumerate(self.trainloader, 0):
            # get the inputs
            if len(data) == 2:
                inputs, labels = data
                tasks = None
            else:
                inputs, tasks, labels = data

            if self.cuda is not None:
                inputs = inputs.cuda(self.cuda)
                labels = labels.cuda(self.cuda)

            task_labels = [torch.tensor([self.tasks[t].label_to_task(l) for l in labels], dtype=torch.long) for t in range(self.active_tasks)]

            if self.cuda is not None:
                task_labels = [t.cuda(self.cuda) for t in task_labels]

            # zero the parameter gradients
            self.optimizer.zero_grad()

            # get non-gate parameter loss
            losses = self.get_losses(inputs, task_labels, tasks=tasks, sample=sample, naive_sample=naive_sample, individual=individual, mixture=mixture, VI=VI)

            scales = [t.scale for t in self.tasks]

            net_loss = sum([losses[i] * scales[i] for i in range(len(self.tasks))]) / len(self.tasks)

            #get gate loss
            loss = net_loss
            loss.backward()
            self.optimizer.step()

            # print statistics
            with torch.no_grad():
                running_loss_total += loss.item()
                if i % log_interval == log_interval-1:
                    if self.log:
                        print('[%5d] loss: %.3f' % (i + 1, running_loss_total / log_interval), file=open(self.log, "a"))
                    else:
                        print('[%5d] loss: %.3f' % (i + 1, running_loss_total / log_interval))
                    running_loss_total = 0.0

                    for name, param in self.named_parameters():
                        if param.requires_grad:
                            if not individual:
                                if name in ["g_logits1", "g_logits2", "g_logits3"]:
                                    if self.log:
                                        print(name, file=open(self.log, "a"))
                                        for i in range(len(param.data)):
                                            print(list(nn.functional.softmax(param.data[i], dim=0)), file=open(self.log, "a"))
                                        #for i in range(len(self.tasks)):
                                        #    print("grad:", self.gate_log[i] / self.gate_count_log[i], file=open(self.log, "a"))
                                    print(name)
                                    for i in range(len(param.data)):
                                        print(list(nn.functional.softmax(param.data[i], dim=0)))
                                    #for i in range(len(self.tasks)):
                                    #    print("grad:", self.gate_log[i] / self.gate_count_log[i])

                                    self.gate_log.zero_()
                                    self.gate_count_log.zero_()
                            if VI:
                                if name in ["q_logits1", "q_logits2", "q_logits3"]:
                                    if self.log:
                                        print(name, file=open(self.log, "a"))
                                        for i in range(len(param.data)):
                                            print(list(nn.functional.softmax(param.data[i], dim=1)), file=open(self.log, "a"))
                                        #for i in range(len(self.tasks)):
                                        #    print("grad:", self.gate_log[i] / self.gate_count_log[i], file=open(self.log, "a"))
                                    print(name)
                                    for i in range(len(param.data)):
                                        print(list(nn.functional.softmax(param.data[i], dim=1)))

        print('Finished Training')
        if self.log:
            print("Finished Training", file=open(self.log, "a"))


class OneLayerNet(GatingNet):
    def __init__(self, tasks, trainloader, testloader, fc_out, in_channels, log=None, cuda=None, conv_noise=False, gate_noise=False, blocks=3):
        self.active_tasks = 1
        super(OneLayerNet, self).__init__(cuda=cuda, tasks=tasks, blocks=blocks)
        self.trainloader = trainloader
        self.testloader = testloader
        self.log = log
        self.fc_out = fc_out
        self.conv_noise = conv_noise
        self.gate_noise = gate_noise

        """
        self.convolutions = nn.ModuleList([nn.Sequential(
            nn.Conv2d(in_channels, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(128, affine=False),
            Flatten()
        ) for _ in range(self.blocks)])

        """
        self.convolutions = nn.ModuleList([nn.Sequential(
            SimpleConvNetBlock(in_channels, 32, 3),
            SimpleConvNetBlock(32, 32, 3),
            SimpleConvNetBlock(32, 32, 3),
            SimpleConvNetBlock(32, 32, 3),
            nn.BatchNorm2d(32),
            Flatten()
        ) for _ in range(self.blocks)])



        self.fc_layer1 = nn.ModuleList([nn.Sequential(
            nn.Linear(self.fc_out, 16),
            nn.ReLU()
        ) for _ in range(len(self.tasks))])

        """
        self.fc_layer2 = nn.ModuleList([nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU()
        ) for _ in range(len(self.tasks))])
        """

        self.fc_layer3 = nn.ModuleList([nn.Sequential(
            nn.Linear(16, tasks[i].size)
        ) for i in range(len(self.tasks))])

        self.g_logits1 = nn.Parameter(torch.zeros([len(self.tasks), self.blocks]))
        self.q_logits1 = nn.Parameter(torch.zeros([len(self.tasks), 5, self.blocks]))

        self.setup(layered=True)

    def forward(self, img, tasks=None, test=False, sample=False, naive_sample=False, mixture=False,
                individual=False,
                VI=False, labels=None):
        blocks = torch.stack([self.convolutions[i](img) for i in range(len(self.convolutions))], dim=0)

        if self.training:
            if not (sample or naive_sample or mixture or VI or individual):
                stack = torch.stack([t for t in blocks], dim=1)
                gates = [nn.functional.softmax(self.g_logits1[i], dim=0) for i in range(len(self.g_logits1))]
                blocks = torch.stack([(stack * g.view(-1, 1)).sum(dim=1) for g in gates])

            for layer in [self.fc_layer1]:
                if individual or not (sample or naive_sample or VI):
                    blocks = torch.stack([layer[i](blocks[i]) for i in range(len(layer))], dim=0)
                else:
                    blocks = torch.stack([layer[i](blocks) for i in range(len(layer))], dim=0)

            for layer in [self.fc_layer3]:
                blocks = torch.stack([layer[i](blocks[i]) for i in range(len(layer))], dim=0)


        else:
            if not (sample or naive_sample or mixture or VI or individual):
                stack = torch.stack([t for t in blocks], dim=1)
                gates = [nn.functional.softmax(self.g_logits1[i], dim=0) for i in range(len(self.g_logits1))]
                blocks = torch.stack([(stack * g.view(-1, 1)).sum(dim=1) for g in gates])
            elif not individual:
                #dim1 = torch.distributions.OneHotCategorical(logits=self.g_logits1).sample([blocks.shape[1]])
                dim1 = torch.argmax(self.g_logits1, dim=1, keepdim=True)
                mask = torch.zeros(self.g_logits1.shape[0], self.g_logits1.shape[1])
                if self.cuda is not None:
                    mask = mask.cuda(self.cuda)

                dim1 = mask.scatter_(1, dim1, 1)

                blocks = blocks[None]
                blocks = blocks * dim1[:, :, None, None]
                blocks = blocks.sum(1)

            for layer in [self.fc_layer1, self.fc_layer3]:
                blocks = torch.stack([layer[i](blocks[i]) for i in range(len(layer))], dim=0)

        return blocks

    def get_losses(self, inputs, task_labels, tasks=None, naive_sample=False, sample=False, individual=False,
                   mixture=False, VI=False):
        outputs = self(inputs, naive_sample=naive_sample, sample=sample, individual=individual,
                       mixture=mixture, VI=VI)

        losses = torch.zeros(len(self.tasks))

        if self.cuda is not None:
            losses = losses.cuda(self.cuda)

        if naive_sample:

            soft_matrix = F.softmax(self.g_logits1, dim=1)
            #put the batch size, answer first N x C x d1 x d2
            outputs = outputs.permute(2, 3, 0, 1)

            #N x d1 x d2
            labels = torch.stack(task_labels, dim=1)[:,:,None].expand(-1, -1, self.blocks)

            if tasks is not None:
                loss_mat = (self.criterion(outputs, labels) * soft_matrix).sum(dim=2)
                loss_mask = torch.zeros(loss_mat.shape[0], loss_mat.shape[1])
                if self.cuda is not None:
                    loss_mask = loss_mask.cuda(self.cuda)

                loss_mask = loss_mask.scatter_(1, tasks.reshape(-1, 1), 1)
                losses += (loss_mat * loss_mask).mean(0)
            else:
                loss = (self.criterion(outputs, labels).mean(dim=0) * soft_matrix).sum(dim=1)
                losses += loss

        elif sample:

            soft_log_matrix = F.log_softmax(self.g_logits1, dim=1)

            # put the batch size, answer first N x C x d1 x d2
            outputs = outputs.permute(2, 3, 0, 1)

            # N x d1 x d2
            labels = torch.stack(task_labels, dim=1)[:, :, None].expand(-1, -1, self.blocks)

            if tasks is not None:
                loss_mat = (-(-self.criterion(outputs, labels) + soft_log_matrix).logsumexp(dim=2))
                loss_mask = torch.zeros(loss_mat.shape[0], loss_mat.shape[1])
                if self.cuda is not None:
                    loss_mask = loss_mask.cuda(self.cuda)

                loss_mask = loss_mask.scatter_(1, tasks.reshape(-1, 1), 1)
                losses += (loss_mat * loss_mask).mean(0)
            else:
                loss = (-(-self.criterion(outputs, labels) + soft_log_matrix).logsumexp(dim=2)).mean(0)
                losses += loss

        elif VI:

            log_p_matrix = F.log_softmax(self.g_logits1, dim=1)
            log_q_matrix = F.log_softmax(self.q_logits1, dim=2)
            q_matrix = F.softmax(self.q_logits1, dim=2)

            task_labels = torch.stack(task_labels, dim=1)
            log_q_matrix = log_q_matrix[None].expand(task_labels.shape[0], -1, -1, -1)
            log_q_matrix = torch.gather(log_q_matrix, 2, task_labels[:, :, None, None].expand(-1, -1, 1, self.blocks)).sum(2)

            q_matrix = q_matrix[None].expand(task_labels.shape[0], -1, -1, -1)
            q_matrix = torch.gather(q_matrix, 2, task_labels[:, :, None, None].expand(-1, -1, 1, self.blocks)).sum(2)

            # put the batch size, answer first N x C x d1 x d2
            outputs = outputs.permute(2, 3, 0, 1)

            # N x d1 x d2
            labels = task_labels[:, :, None].expand(-1, -1, self.blocks)

            if tasks is not None:
                loss_mat = ((self.criterion(outputs, labels) - log_p_matrix + log_q_matrix) * q_matrix).sum(dim=2)
                loss_mask = torch.zeros(loss_mat.shape[0], loss_mat.shape[1])
                if self.cuda is not None:
                    loss_mask = loss_mask.cuda(self.cuda)

                loss_mask = loss_mask.scatter_(1, tasks.reshape(-1, 1), 1)
                losses += (loss_mat * loss_mask).mean(0)
            else:
                loss = ((self.criterion(outputs, labels) - log_p_matrix + log_q_matrix) * q_matrix).mean(dim=0).sum(dim=1)
                losses += loss

        else:  # blend/individual
            outputs = outputs.permute(1, 2, 0)
            labels = torch.stack(task_labels, dim=1)

            if tasks is not None:
                loss_mat = self.criterion(outputs, labels)
                loss_mask = torch.zeros(loss_mat.shape[0], loss_mat.shape[1])
                if self.cuda is not None:
                    loss_mask = loss_mask.cuda(self.cuda)

                loss_mask = loss_mask.scatter_(1, tasks.reshape(-1, 1), 1)
                losses += (loss_mat * loss_mask).mean(0)
            else:
                losses += self.criterion(outputs, labels).mean(0)

        return losses




class SampledOneLayerNet(GatingNet):
    def __init__(self, tasks, trainloader, testloader, fc_out, in_channels, log=None, cuda=None, conv_noise=False, gate_noise=False, blocks=3):
        self.active_tasks = 1
        super(SampledOneLayerNet, self).__init__(cuda=cuda, tasks=tasks, blocks=blocks)
        self.trainloader = trainloader
        self.testloader = testloader
        self.log = log
        self.fc_out = fc_out
        self.conv_noise = conv_noise
        self.gate_noise = gate_noise

        self.convolutions = nn.ModuleList([nn.Sequential(
            SimpleConvNetBlock(in_channels, 32, 3),
            SimpleConvNetBlock(32, 32, 3),
            SimpleConvNetBlock(32, 32, 3),
            SimpleConvNetBlock(32, 32, 3),
            nn.BatchNorm2d(32),
            Flatten()
        ) for _ in range(self.blocks)])



        self.fc_layer1 = nn.ModuleList([nn.Sequential(
            nn.Linear(self.fc_out, 16),
            nn.ReLU()
        ) for _ in range(len(self.tasks))])

        self.fc_layer3 = nn.ModuleList([nn.Sequential(
            nn.Linear(16, tasks[i].size)
        ) for i in range(len(self.tasks))])

        self.g_logits1 = nn.Parameter(torch.zeros([len(self.tasks), self.blocks]))
        self.q_logits1 = nn.Parameter(torch.zeros([len(self.tasks), 5, self.blocks]))

        self.setup(layered=True)

    def forward(self, img, tasks=None, test=False, sample=False, naive_sample=False, mixture=False,
                individual=False,
                VI=False, labels=None):

        if self.training:
            dim1 = torch.distributions.Categorical(logits=self.g_logits1).sample([img.shape[0], self.num_samples])
            dim1 = dim1.permute(0, 2, 1) #batch_s x tasks x samples
        else:
            dim1 = torch.argmax(self.g_logits1, dim=1, keepdim=True)

        b_map = {}
        for b_index in torch.unique(dim1):
            b_map[int(b_index)] = self.convolutions[b_index](img)


        #how can you do this in pytorch without looping which is suboptimal???


        for layer in [self.fc_layer1, self.fc_layer3]:
            blocks = torch.stack([layer[i](blocks[i]) for i in range(len(layer))], dim=0)

        return blocks

    def get_losses(self, inputs, task_labels, tasks=None, naive_sample=False, sample=False, individual=False,
                   mixture=False, VI=False):
        outputs = self(inputs, naive_sample=naive_sample, sample=sample, individual=individual,
                       mixture=mixture, VI=VI)

        losses = torch.zeros(len(self.tasks))
        log_probs = torch.zeros(32, self.num_samples)

        if self.cuda is not None:
            losses = losses.cuda(self.cuda)
            log_probs = log_probs.cuda(self.cuda)

        if naive_sample:

            soft_matrix = F.softmax(self.g_logits1, dim=1)
            #put the batch size, answer first N x C x d1 x d2
            outputs = outputs.permute(2, 3, 0, 1)

            #N x d1 x d2
            labels = torch.stack(task_labels, dim=1)[:,:,None].expand(-1, -1, self.blocks)

            if tasks is not None:
                loss_mat = (self.criterion(outputs, labels) * soft_matrix).sum(dim=2)
                loss_mask = torch.zeros(loss_mat.shape[0], loss_mat.shape[1])
                if self.cuda is not None:
                    loss_mask = loss_mask.cuda(self.cuda)

                loss_mask = loss_mask.scatter_(1, tasks.reshape(-1, 1), 1)
                losses += (loss_mat * loss_mask).mean(0)
            else:
                loss = (self.criterion(outputs, labels).mean(dim=0) * soft_matrix).sum(dim=1)
                losses += loss

        elif sample:

            soft_log_matrix = F.log_softmax(self.g_logits1, dim=1)

            # put the batch size, answer first N x C x d1 x d2
            outputs = outputs.permute(2, 3, 0, 1)

            # N x d1 x d2
            labels = torch.stack(task_labels, dim=1)[:, :, None].expand(-1, -1, self.blocks)

            if tasks is not None:
                loss_mat = (-(-self.criterion(outputs, labels) + soft_log_matrix).logsumexp(dim=2))
                loss_mask = torch.zeros(loss_mat.shape[0], loss_mat.shape[1])
                if self.cuda is not None:
                    loss_mask = loss_mask.cuda(self.cuda)

                loss_mask = loss_mask.scatter_(1, tasks.reshape(-1, 1), 1)
                losses += (loss_mat * loss_mask).mean(0)
            else:
                loss = (-(-self.criterion(outputs, labels) + soft_log_matrix).logsumexp(dim=2)).mean(0)
                losses += loss

        elif VI:
            log_p_matrix = F.log_softmax(self.g_logits1, dim=1)
            log_q_matrix = F.log_softmax(self.q_logits1, dim=2)
            q_matrix = F.softmax(self.q_logits1, dim=2)

            task_labels = torch.stack(task_labels, dim=1)
            log_q_matrix = log_q_matrix[None].expand(task_labels.shape[0], -1, -1, -1)
            log_q_matrix = torch.gather(log_q_matrix, 2, task_labels[:, :, None, None].expand(-1, -1, 1, self.blocks)).sum(2)

            q_matrix = q_matrix[None].expand(task_labels.shape[0], -1, -1, -1)
            q_matrix = torch.gather(q_matrix, 2, task_labels[:, :, None, None].expand(-1, -1, 1, self.blocks)).sum(2)

            # put the batch size, answer first N x C x d1 x d2
            outputs = outputs.permute(2, 3, 0, 1)

            # N x d1 x d2
            labels = task_labels[:, :, None].expand(-1, -1, self.blocks)

            if tasks is not None:
                loss_mat = ((self.criterion(outputs, labels) - log_p_matrix + log_q_matrix) * q_matrix).sum(dim=2)
                loss_mask = torch.zeros(loss_mat.shape[0], loss_mat.shape[1])
                if self.cuda is not None:
                    loss_mask = loss_mask.cuda(self.cuda)

                loss_mask = loss_mask.scatter_(1, tasks.reshape(-1, 1), 1)
                losses += (loss_mat * loss_mask).mean(0)
            else:
                loss = ((self.criterion(outputs, labels) - log_p_matrix + log_q_matrix) * q_matrix).mean(dim=0).sum(dim=1)
                losses += loss

        return losses