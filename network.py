import torch
from torch import nn
import torch.optim as optim
import numpy as np
from gated_layer import SamplingLayer, BlendingLayer, VILayer


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size()[0], -1)


class SimpleConvNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        y = self.conv(x)
        y = self.relu(y)
        y = self.maxpool(y)
        return y


class QNet(nn.Module):
    def __init__(self, block, flatten_size, num_blocks):
        super().__init__()
        self.block = block
        self.fc = nn.Linear(flatten_size, num_blocks)

    def forward(self, x, concat):
        x = self.block(x)
        x = torch.cat((x, concat), 1)
        out = self.fc(x)
        return out


class Network(nn.Module):
    def __init__(self, tasks, trainloader, validloader, testloader, config=None, log=None, cuda=None):
        super(Network, self).__init__()
        self.cuda = cuda
        self.tasks = tasks
        self.trainloader = trainloader
        self.validloader = validloader
        self.testloader = testloader
        self.reg = config["reg"]
        self.in_channels = config["in_channels"]
        self.blocks = config["blocks"]
        self.fc_in = config["fc_in"]
        self.options = config["options"]
        self.log = log
        self.config = config
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        if self.cuda:
            self.criterion.cuda(self.cuda)

    def setup(self):
        self.optimizer = optim.Adam(self.parameters())

    def train_epoch(self, log_interval=1000):
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

            tasks = tasks[:, None]

            task_labels = torch.stack([torch.tensor(self.tasks[tasks[l]].label_to_task(labels[l]), dtype=torch.long) for l in range(len(labels))])
            task_labels = task_labels[:, None]

            if self.cuda is not None:
                task_labels = task_labels.cuda(self.cuda)

            # zero the parameter gradients
            self.optimizer.zero_grad()

            # get non-gate parameter loss
            losses, ans_loss, gate_loss = self.get_losses(inputs, tasks, task_labels)
            losses.backward()
            self.optimizer.step()

            # print statistics
            with torch.no_grad():
                if isinstance(gate_loss, torch.Tensor):
                    gate_loss = gate_loss.item()
                running_loss_total += ans_loss.item() + gate_loss
                if i % log_interval == log_interval - 1:
                    if self.log:
                        print('[%5d] loss: %.3f' % (i + 1, running_loss_total / log_interval), file=open(self.log, "a"))
                        print('[%5d] loss: %.3f' % (i + 1, running_loss_total / log_interval))
                    else:
                        print('[%5d] loss: %.3f' % (i + 1, running_loss_total / log_interval))
                    running_loss_total = 0.0
                    self.log_gating()

        print('Finished Training')
        if self.log:
            print("Finished Training", file=open(self.log, "a"))

    def log_gating(self):
        raise NotImplementedError

    def test_overall(self):
        self.eval()

        correct_pos = 0
        total_pos = 0

        correct_neg = 0
        total_neg = 0

        total = 0
        correct = 0

        running_reg_loss = 0
        running_answer_loss = 0
        running_answer_reg_loss = 0
        count = 0

        with torch.no_grad():

            for data in self.validloader:

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

                tasks = tasks[:, None]
                task_labels = torch.stack(
                    [torch.tensor(self.tasks[tasks[l]].label_to_task(labels[l]), dtype=torch.long) for l in
                     range(len(labels))])
                task_labels = task_labels[:, None]

                if self.cuda is not None:
                    task_labels = task_labels.cuda(self.cuda)

                _, answer_loss, reg_loss = self.get_losses(inputs, tasks, task_labels)
                answer_reg_loss = answer_loss + reg_loss
                running_answer_loss += answer_loss
                running_reg_loss += reg_loss
                running_answer_reg_loss += answer_reg_loss
                count += 1

                outputs, _, _ = self.forward(inputs, tasks, None)
                predicted = torch.argmax(outputs, dim=1)
                task_labels = task_labels.reshape(-1)

                # True Positive (TP): we predict a label of 1 (positive), and the true label is 1.
                TP = np.logical_and(task_labels.cpu() == 1, predicted.cpu() == 1).sum()

                # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
                TN = np.logical_and(task_labels.cpu() == 0, predicted.cpu() == 0).sum()

                correct_pos += float(TP)
                correct_neg += float(TN)
                total_pos += float(sum(task_labels == 1))
                total_neg += float(sum(task_labels == 0))

                correct += float(sum(task_labels == predicted))
                total += len(task_labels)

            if self.log:
                print('overall accuracy of the network on the 10000 valid images: %.2f %%' % (
                        100 * float(correct) / float(total)), file=open(self.log, "a"))
                print('overall valid loss of the network: pred: %.2f, gate: %.2f, pred+gate: %.2f' %
                      (running_answer_loss/count, running_reg_loss/count, running_answer_reg_loss/count),
                      file=open(self.log, "a"))

            print('overall accuracy of the network on the 10000 valid images: %.2f %%' % (
                    100 * float(correct) / float(total)))
            print('overall valid loss of the network: pred: %.2f, gate: %.2f, pred+gate: %.2f' %
                  (running_answer_loss/count, running_reg_loss/count, running_answer_reg_loss/count))

            correct_pos = 0
            total_pos = 0

            correct_neg = 0
            total_neg = 0

            total = 0
            correct = 0

            running_reg_loss = 0
            running_answer_loss = 0
            running_answer_reg_loss = 0
            count = 0

            for data in self.testloader:

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

                tasks = tasks[:, None]
                task_labels = torch.stack(
                    [torch.tensor(self.tasks[tasks[l]].label_to_task(labels[l]), dtype=torch.long) for l in range(len(labels))])
                task_labels = task_labels[:, None]
                if self.cuda is not None:
                    task_labels = task_labels.cuda(self.cuda)

                _, answer_loss, reg_loss = self.get_losses(inputs, tasks, task_labels)
                answer_reg_loss = answer_loss + reg_loss
                running_answer_loss += answer_loss
                running_reg_loss += reg_loss
                running_answer_reg_loss += answer_reg_loss

                outputs, _, _ = self.forward(inputs, tasks, None)
                predicted = torch.argmax(outputs, dim=1)
                task_labels = task_labels.reshape(-1)

                # True Positive (TP): we predict a label of 1 (positive), and the true label is 1.
                TP = np.logical_and(task_labels.cpu() == 1, predicted.cpu() == 1).sum()

                # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
                TN = np.logical_and(task_labels.cpu() == 0, predicted.cpu() == 0).sum()

                correct_pos += float(TP)
                correct_neg += float(TN)
                total_pos += float(sum(task_labels == 1))
                total_neg += float(sum(task_labels == 0))
                count += 1

                correct += float(sum(task_labels == predicted))
                total += len(task_labels)

            if self.log:
                print('overall accuracy of the network on the 10000 test images: %.2f %%' % (
                        100 * float(correct) / float(total)), file=open(self.log, "a"))
                print('overall test loss of the network: pred: %.2f, gate: %.2f, pred+gate: %.2f' %
                      (running_answer_loss/count, running_reg_loss/count, running_answer_reg_loss/count), file=open(self.log, "a"))

            print('overall accuracy of the network on the 10000 test images: %.2f %%' % (
                        100 * float(correct) / float(total)))
            print('overall test loss of the network: pred: %.2f, gate: %.2f, pred+gate: %.2f' %
                  (running_answer_loss/count, running_reg_loss/count, running_answer_reg_loss/count))

        self.train()


class Individual(Network):
    def __init__(self, tasks, trainloader, validloader, testloader, config=None, log=None, cuda=None):
        super().__init__(tasks, trainloader, validloader, testloader, config=config, log=log, cuda=cuda)

        self.convs = nn.ModuleList([nn.Sequential(
            SimpleConvNetBlock(self.in_channels, 32, 3),
            SimpleConvNetBlock(32, 32, 3),
            SimpleConvNetBlock(32, 32, 3),
            SimpleConvNetBlock(32, 32, 3),
            nn.BatchNorm2d(32),
            Flatten()
        ) for _ in range(self.blocks)])

        self.fcs = nn.ModuleList([nn.Sequential(
            nn.Linear(self.fc_in, 16),
            nn.Linear(16, self.tasks[i].size)
        ) for i in range(len(self.tasks))])

        self.setup()

    def log_gating(self):
        return

    def forward(self, input, emb, labels):
        """
        :param input: Nximg_size
        :param emb: Nx1
        :param labels: Nx1
        :return: out: NxO, log_qs: NxB, log_ps: NxB
        """
        N = input.shape[0]

        out = torch.stack([self.convs[i](input) for i in range(len(self.convs))])
        out = torch.stack([self.fcs[i](out[i]) for i in range(len(self.convs))])

        O = out.shape[-1]

        out = torch.gather(out, 0, emb[None, :, :].expand(1, -1, self.options))
        out = out.reshape(N, O)

        return out, 0, 0

    def get_losses(self, inputs, emb, labels):
        results, _, _ = self(inputs, emb, labels)
        labels = labels.view(-1)
        losses = self.criterion(results, labels).mean(dim=0)
        return losses, losses, 0


class OneLayer(Network):
    def __init__(self, tasks, trainloader, validloader, testloader, config=None, log=None, cuda=None):
        super().__init__(tasks, trainloader, validloader, testloader, config=config, log=log, cuda=cuda)

        self.convs = nn.ModuleList([nn.Sequential(
            SimpleConvNetBlock(self.in_channels, 32, 3),
            SimpleConvNetBlock(32, 32, 3),
            SimpleConvNetBlock(32, 32, 3),
            SimpleConvNetBlock(32, 32, 3),
            nn.BatchNorm2d(32),
            Flatten()
        ) for _ in range(self.blocks)])

        """
        self.conv = nn.Sequential(
            SimpleConvNetBlock(self.in_channels, 32, 3),
            SimpleConvNetBlock(32, 32, 3),
            SimpleConvNetBlock(32, 32, 3),
            SimpleConvNetBlock(32, 32, 3),
            nn.BatchNorm2d(32),
            Flatten()
        )

        self.fc_blocks = nn.ModuleList([nn.Sequential(
            nn.Linear(self.fc_in, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        ) for i in range(self.blocks)])
        """

        self.fcs = nn.ModuleList([nn.Sequential(
            nn.Linear(self.fc_in, 16),
            nn.Linear(16, self.tasks[i].size)
        ) for i in range(len(self.tasks))])

        self.q_block = nn.Sequential(
            SimpleConvNetBlock(self.in_channels, 32, 3),
            SimpleConvNetBlock(32, 32, 3),
            Flatten()
        )

        self.q_net = QNet(self.q_block, 2568, self.blocks)

    def forward(self, input, emb, labels):
        """
        :param input: Nximg_size
        :param emb: Nx1
        :param labels: Nx1
        :return: out: NxO, log_probs: Nx1, extra_loss: Nx1
        """
        N = input.shape[0]

        log_probs = 0.0
        extra_loss = 0.0
        for i in range(len(self.gated_layers)):
            input, log_probs, extra_loss = self.gated_layers[i](input, emb, log_probs=log_probs, extra_loss=extra_loss, labels=labels)

        out = torch.stack([f(input) for f in self.fcs])

        O = out.shape[-1]

        out = torch.gather(out, 0, emb[None, :, :].expand(1, -1, self.options))
        out = out.reshape(N, O)

        return out, log_probs, extra_loss

    def get_losses(self, inputs, emb, labels):
        results, log_probs, extra_loss = self(inputs, emb, labels)
        labels = labels.reshape(-1)

        avg_extra_loss = 0
        if isinstance(log_probs, torch.Tensor):
            log_probs = log_probs.reshape(-1, 1)
        if isinstance(extra_loss, torch.Tensor):
            extra_loss = extra_loss.reshape(-1, 1)
            avg_extra_loss = extra_loss.mean(dim=0)

        loss = self.criterion(results, labels).reshape(-1, 1)

        return (loss + (loss.detach() * log_probs) + extra_loss).mean(dim=0), loss.mean(dim=0), avg_extra_loss


class OneLayerBlend(OneLayer):
    def __init__(self, tasks, trainloader, validloader, testloader, config=None, log=None, cuda=None):
        super().__init__(tasks, trainloader, validloader, testloader, config=config, log=log, cuda=cuda)
        self.gated_layers = nn.ModuleList([BlendingLayer(self.convs, self.config, len(tasks), cuda=self.cuda)])
        self.setup()

    def log_gating(self):
        for layer in self.gated_layers:
            for name, param in layer.named_parameters():
                if param.requires_grad:
                    if name in ["g_logits"]:
                        if self.log:
                            print(name, file=open(self.log, "a"))
                            for i in range(len(param.data)):
                                print(list(param.data[i]),
                                      file=open(self.log, "a"))
                        print(name)
                        for i in range(len(param.data)):
                            print(list(torch.sigmoid(param.data[i])))

                    if name in ["weights"]:
                        if self.log:
                            print(name, file=open(self.log, "a"))
                            for i in range(len(param.data)):
                                print(list(param.data[i]),
                                      file=open(self.log, "a"))
                        print(name)
                        for i in range(len(param.data)):
                            print(list(param.data[i]))

class OneLayerSample(OneLayer):
    def __init__(self, tasks, trainloader, validloader, testloader, config=None, log=None, cuda=None):
        super().__init__(tasks, trainloader, validloader, testloader, config=config, log=log, cuda=cuda)
        self.gated_layers = nn.ModuleList([SamplingLayer(self.convs, self.config, len(tasks), out_shape=config["fc_in"], cuda=self.cuda)])
        self.setup()

    def log_gating(self):
        for layer in self.gated_layers:
            for name, param in layer.named_parameters():
                if param.requires_grad:
                    if name in ["g_logits"]:
                        if self.log:
                            print(name, file=open(self.log, "a"))
                            for i in range(len(param.data)):
                                print(list(torch.sigmoid(param.data[i])),
                                      file=open(self.log, "a"))
                        print(name)
                        for i in range(len(param.data)):
                            print(list(torch.sigmoid(param.data[i])))

                    if name in ["weights"]:
                        if self.log:
                            print(name, file=open(self.log, "a"))
                            for i in range(len(param.data)):
                                print(list(param.data[i]),
                                      file=open(self.log, "a"))
                        print(name)
                        for i in range(len(param.data)):
                            print(list(param.data[i]))

class OneLayerVI(OneLayer):
    def __init__(self, tasks, trainloader, validloader, testloader, config=None, log=None, cuda=None):
        super().__init__(tasks, trainloader, validloader, testloader, config=config, log=log, cuda=cuda)
        self.gated_layers = nn.ModuleList([VILayer(self.convs, self.config, len(tasks), self.q_net, out_shape=config["fc_in"], cuda=self.cuda)])
        self.setup()

    def log_gating(self):
        for layer in self.gated_layers:
            for name, param in layer.named_parameters():
                if param.requires_grad:
                    if name in ["g_logits"]:
                        if self.log:
                            print(name, file=open(self.log, "a"))
                            for i in range(len(param.data)):
                                print(list(torch.sigmoid(param.data[i])),
                                      file=open(self.log, "a"))
                        print(name)
                        for i in range(len(param.data)):
                            print(list(torch.sigmoid(param.data[i])))

                    if name in ["weights"]:
                        if self.log:
                            print(name, file=open(self.log, "a"))
                            for i in range(len(param.data)):
                                print(list(param.data[i]),
                                      file=open(self.log, "a"))
                        print(name)
                        for i in range(len(param.data)):
                            print(list(param.data[i]))

