import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import normal


class GatingNet(nn.Module):
    def __init__(self, cuda=None, tasks=None):
        super(GatingNet, self).__init__()
        self.cuda = cuda
        self.blocks = 3
        self.normal = normal.Normal(0.0, 0.0001)
        self.tasks = tasks
        self.gate_log = torch.zeros([len(self.tasks), self.blocks])
        self.gate_count_log = torch.zeros([len(self.tasks)])

        if cuda:
            self.gate_log = self.gate_log.cuda(cuda)
            self.gate_count_log = self.gate_count_log.cuda(cuda)

    def setup(self, optimizer=None, criterion=None):
        self.g_logits = nn.Parameter(torch.zeros([len(self.tasks), self.blocks]))
        self.config = torch.zeros(self.blocks, len(self.tasks), 1, self.blocks, 1)
        for c in range(self.blocks):
            for t in range(len(self.tasks)):
                for b in range(self.blocks):
                    if c == b:
                        self.config[c, t, 0, b, 0] = 1


        """
        self.mix_logits = nn.Parameter(torch.zeros([len(self.tasks), 2 ** self.blocks]))
        self.mix_config = torch.zeros(2 ** self.blocks, len(self.tasks), 1, self.blocks, 1)
        for c in range(2 ** self.blocks):
            for t in range(len(self.tasks)):
                for b in range(self.blocks):
                    if (c // (2**b)) % 2 == 1:
                        self.mix_config[c, t, 0, b, 0] = 1

        self.mix_map = torch.zeros(2 ** self.blocks, self.blocks)
        for c in range(2 ** self.blocks):
            for b in range(self.blocks):
                if (c // (2 ** b)) % 2 == 1:
                    self.mix_map[c, b] = 1
        """

        if self.cuda:
            self.config = self.config.cuda(self.cuda)
            self.mix_config = self.config.cuda(self.cuda)

        self.softmax = nn.LogSoftmax(dim=1)
        if optimizer is None:
            self.optimizer = optim.Adam(self.parameters())
        if criterion is None:
            self.criterion = nn.CrossEntropyLoss()
            if self.cuda:
                self.criterion.cuda(self.cuda)

    def train_epoch(self, sample=False, individual=False, mixture=False, log_interval=1000):
        print("Starting Training")
        if self.log:
            print("Starting Training", file=open(self.log, "a"))
        running_loss_total = 0.0
        for i, data in enumerate(self.trainloader, 0):
            # get the inputs
            inputs, labels = data
            if self.cuda is not None:
                inputs = inputs.cuda(self.cuda)
                labels = labels.cuda(self.cuda)

            task_labels = [torch.tensor([t.label_to_task(l) for l in labels], dtype=torch.long) for t in self.tasks]

            if self.cuda is not None:
                task_labels = [t.cuda(self.cuda) for t in task_labels]

            # zero the parameter gradients
            self.optimizer.zero_grad()

            # get non-gate parameter loss
            if sample: #sample
                outputs, probs = self.forward(inputs, sample=sample, individual=individual, mixture=mixture)
                losses = [torch.stack([self.criterion(outputs[t][c], task_labels[t]) * probs[t][c] for c in range(self.blocks)]).sum()
                          for t in range(len(self.tasks))]

            elif mixture:
                outputs, probs = self.forward(inputs, sample=sample, individual=individual, mixture=mixture)
                losses = [torch.stack(
                    [self.criterion(outputs[t][c], task_labels[t]) * probs[t][c] for c in range(2 ** self.blocks)]).sum()
                          for t in range(len(self.tasks))]
            else: #blend/individual
                outputs = self.forward(inputs, sample=sample, individual=individual, mixture=mixture)
                losses = [self.criterion(out, label) for out, label in zip(outputs, task_labels)]

            scales = [t.scale for t in self.tasks]
            net_loss = sum([losses[i] * scales[i] for i in range(len(self.tasks))]) / len(losses)

            #get gate loss
            #gate_loss = net_loss.detach() * probs
            loss = net_loss #+ gate_loss
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
                            if name == "g_logits":
                                if self.log:
                                    print(name, param.data, file=open(self.log, "a"))
                                    for i in range(len(self.tasks)):
                                        print(nn.functional.softmax(param.data[i], dim=0), file=open(self.log, "a"))
                                    #for i in range(len(self.tasks)):
                                    #    print("grad:", self.gate_log[i] / self.gate_count_log[i], file=open(self.log, "a"))
                                print(name, param.data)
                                for i in range(len(self.tasks)):
                                    print(nn.functional.softmax(param.data[i], dim=0))
                                #for i in range(len(self.tasks)):
                                #    print("grad:", self.gate_log[i] / self.gate_count_log[i])

                                self.gate_log.zero_()
                                self.gate_count_log.zero_()
                            if name == "mix_logits":
                                if self.log:
                                    print(name, param.data, file=open(self.log, "a"))
                                    for i in range(len(self.tasks)):
                                        print("mix:", nn.functional.softmax(param.data[i], dim=0), file=open(self.log, "a"))

                                print(name, param.data)
                                for i in range(len(self.tasks)):
                                    print("mix:", nn.functional.softmax(param.data[i], dim=0))

        print('Finished Training')
        if self.log:
            print("Finished Training", file=open(self.log, "a"))

    def test_overall(self, sample=False, individual=False, mixture=False):
        self.eval()
        correct_arr = [0 for i in range(len(self.tasks))]
        total_arr = [0 for i in range(len(self.tasks))]
        with torch.no_grad():
            for data in self.testloader:
                inputs, labels = data
                if self.cuda != None:
                    inputs = inputs.cuda(self.cuda)
                    labels = labels.cuda(self.cuda)

                task_labels = [torch.tensor([t.label_to_task(l) for l in labels], dtype=torch.long) for t in self.tasks]
                if self.cuda != None:
                    task_labels = [t.cuda(self.cuda) for t in task_labels]

                outputs = self.forward(inputs, test=True, sample=sample, individual=individual, mixture=mixture)
                predicted = [torch.max(output.data, 1)[1] for output in outputs]

                for i in range(len(self.tasks)):
                    total_arr[i] += task_labels[i].size(0)
                    correct_arr[i] += (predicted[i] == task_labels[i]).sum().item()

        if self.log:
            for i in range(len(self.tasks)):
                print('task_' + str(i) + ' accuracy of the network on the 10000 test images: %d %%' % (
                        100 * correct_arr[i] / total_arr[i]), file=open(self.log, "a"))

        for i in range(len(self.tasks)):
            print('task_' + str(i) + ' accuracy of the network on the 10000 test images: %d %%' % (
                    100 * correct_arr[i] / total_arr[i]))

        self.train()

    def log_gate(self, v, i):
        self.gate_log[i] += v

    def add_normal_noise(self, v):
        return v + self.normal.sample()

    def log_assign(self, i):
        return lambda grad: self.log_gate(grad, i)

    def forward(self, img, test=False, sample=False, individual=False, mixture=False):

        blocks = [self.convs[i](img) for i in range(self.blocks)]
        stack = torch.stack([t.view(-1, self.fc_out) for t in blocks], dim=1) #batch_size x tasks x vals

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

        if sample:
            if self.training:
                #print("sample to train")
                stack = stack.reshape(1, stack.shape[0], stack.shape[1], stack.shape[2]).repeat(len(self.tasks), 1, 1, 1)
                probs = nn.functional.softmax(self.g_logits, dim=1)

                fc_ins = (stack * self.config).sum(dim=3) #tasks x stack
                fc_ins = fc_ins.permute(1, 0, 2, 3)

                fc_outs = [self.fcs[i](fc_ins[i]) for i in range(len(self.tasks))]
            else:
                sample_gs = [torch.distributions.OneHotCategorical(logits=self.g_logits[i]).sample(
                    [img.shape[0]]) for i in range(len(self.tasks))]  # tasks x blocks x batch_size
                fc_ins = [(stack * g.reshape(g.shape[0], g.shape[1], 1)).sum(dim=1) for g in sample_gs]
                fc_outs = [self.fcs[i](fc_ins[i]) for i in range(len(self.tasks))]
        elif individual:
            fc_outs = [self.fcs[b](blocks[b]) for b in range(self.blocks)]
        elif mixture:
            if self.training:
                probs = nn.functional.softmax(self.mix_logits, dim=1)

                gates = nn.functional.softmax(self.g_logits, dim=1)
                stack = stack.reshape(1, stack.shape[0], stack.shape[1], stack.shape[2]).repeat(len(self.tasks), 1, 1, 1)
                stack = stack * gates.view(len(self.tasks), 1, self.blocks, 1)
                fc_ins = (stack * self.mix_config).sum(dim=3)  # tasks x stack
                fc_ins = fc_ins.permute(1, 0, 2, 3) #task x config x batch_size x block_size
                fc_outs = [self.fcs[i](fc_ins[i]) for i in range(len(self.tasks))]
            else:
                gates = [nn.functional.softmax(self.g_logits[i], dim=0) for i in range(len(self.tasks))]
                sample_gs = [torch.mm(torch.distributions.OneHotCategorical(logits=self.mix_logits[i]).sample(
                    [img.shape[0]]), self.mix_map) for i in range(len(self.tasks))]  # tasks x 2**blocks x batch_size

                fc_ins = [(stack * sample_gs[i].reshape(sample_gs[i].shape[0], sample_gs[i].shape[1], 1) *
                           gates[i].view(-1, 1)).sum(dim=1) for i in range(len(sample_gs))]
                fc_outs = [self.fcs[i](fc_ins[i]) for i in range(len(self.tasks))]
        else: #blending
            gates = [nn.functional.softmax(self.g_logits[i], dim=0) for i in range(len(self.tasks))]
            fc_ins = [(stack * g.view(-1, 1)).sum(dim=1) for g in gates]
            fc_outs = [self.fcs[i](fc_ins[i]) for i in range(len(self.tasks))]

        #sample and output
        #if test == False:
        #    for i in range(len(gates)):
        #        gates[i].register_hook(self.log_assign(i))
        #        self.gate_count_log[i] += 1

        if (sample or mixture) and self.training:
            return [self.softmax(out) for out in fc_outs], probs
        else:
            return [self.softmax(out) for out in fc_outs]


class LargeNet(GatingNet):
    def __init__(self, tasks, trainloader, testloader, fc_out, log=None, cuda=None, conv_noise=False, gate_noise=False):
        super(LargeNet, self).__init__(cuda=cuda, tasks=tasks)
        self.trainloader = trainloader
        self.testloader = testloader
        self.log = log
        self.fc_out = fc_out
        self.conv_noise = conv_noise
        self.gate_noise = gate_noise

        self.convs = nn.ModuleList([nn.Sequential(
            nn.Conv2d(3, 64, 3),
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
    def __init__(self, tasks, trainloader, testloader, fc_out, log=None, cuda=None, conv_noise=False, gate_noise=False):
        super(LargerNet, self).__init__(cuda=cuda, tasks=tasks)
        self.trainloader = trainloader
        self.testloader = testloader
        self.log = log
        self.fc_out = fc_out
        self.conv_noise = conv_noise
        self.gate_noise = gate_noise

        self.convs = nn.ModuleList([nn.Sequential(
            nn.Conv2d(3, 256, 3),
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
    def __init__(self, tasks, trainloader, testloader, fc_out, log=None, cuda=None, conv_noise=False, gate_noise=False):
        super(AsymmNet, self).__init__(cuda=cuda, tasks=tasks)
        self.trainloader = trainloader
        self.testloader = testloader
        self.log = log
        self.fc_out = fc_out
        self.conv_noise = conv_noise
        self.gate_noise = gate_noise

        self.convs = nn.ModuleList([nn.Sequential(
            nn.Conv2d(3, 64, 3, dilation=2),
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
            nn.Conv2d(3, 64, 5),
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
            nn.Conv2d(3, 64, 3),
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



class SmallNet(GatingNet):
    def __init__(self, tasks, trainloader, testloader, fc_out, log=None, cuda=None, conv_noise=False, gate_noise=False):
        super(SmallNet, self).__init__(cuda=cuda, tasks=tasks)
        self.trainloader = trainloader
        self.testloader = testloader
        self.log = log
        self.fc_out = fc_out
        self.conv_noise = conv_noise
        self.gate_noise = gate_noise

        self.convs = nn.ModuleList([nn.Sequential(
            nn.Conv2d(3, 32, 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        ) for _ in range(3)])

        self.fcs = nn.ModuleList([nn.Sequential(
            nn.Linear(self.fc_out, 16),
            nn.Linear(16, tasks[i].size)
        ) for i in range(len(self.tasks))])

        self.setup()
