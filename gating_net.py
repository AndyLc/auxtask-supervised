import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import normal

class GatingNet(nn.Module):
    def __init__(self, cuda=None):
        super(GatingNet, self).__init__()
        self.cuda = cuda
        self.normal = normal.Normal(0.0, 0.0001)

    def setup(self, optimizer=None, criterion=None):
        if optimizer == None:
            self.optimizer = optim.Adam(self.parameters())
        if criterion == None:
            self.criterion = nn.CrossEntropyLoss()
            if self.cuda:
                self.criterion.cuda(self.cuda)

    def train_epoch(self, disable_non_cifar=False):
        print("Starting Training")
        if self.log:
            print("Starting Training", file=open(self.log, "a"))
        running_loss_total = 0.0
        for i, data in enumerate(self.trainloader, 0):
            # get the inputs
            inputs, labels = data
            if self.cuda != None:
                inputs = inputs.cuda(self.cuda)
                labels = labels.cuda(self.cuda)

            task_labels = [torch.tensor([t.label_to_task(l) for l in labels]) for t in self.tasks]

            if self.cuda != None:
                task_labels = [t.cuda(self.cuda) for t in task_labels]

            # zero the parameter gradients
            self.optimizer.zero_grad()
            losses = [self.criterion(out, label) for out, label in zip(self.forward(inputs), task_labels)]
            if disable_non_cifar:
                loss = losses[3] / 1.5
            else:
                loss = (losses[0] + losses[1] / 3 + losses[2] / 3 + losses[3] * 2) / 4
            loss.backward()
            self.optimizer.step()

            # print statistics
            with torch.no_grad():
                running_loss_total += loss.item()
                if i % 1000 == 999:
                    if self.log:
                        print('[%5d] loss: %.3f' % (i + 1, running_loss_total / 1000), file=open(self.log, "a"))
                    else:
                        print('[%5d] loss: %.3f' % (i + 1, running_loss_total / 1000))
                    running_loss_total = 0.0

                    for name, param in self.named_parameters():
                        if param.requires_grad:
                            if name == "g_logits":
                                if self.log:
                                    print(name, param.data, file=open(self.log, "a"))
                                    print(nn.functional.softmax(param.data[0], dim=0), file=open(self.log, "a"))
                                    print(nn.functional.softmax(param.data[1], dim=0), file=open(self.log, "a"))
                                    print(nn.functional.softmax(param.data[2], dim=0), file=open(self.log, "a"))

                                    print("grad:", self.gate1 / self.gate1_count, file=open(self.log, "a"))
                                    print("grad:", self.gate2 / self.gate2_count, file=open(self.log, "a"))
                                    print("grad:", self.gate4 / self.gate4_count, file=open(self.log, "a"))

                                print(name, param.data)
                                print(nn.functional.softmax(param.data[0], dim=0))
                                print(nn.functional.softmax(param.data[1], dim=0))
                                print(nn.functional.softmax(param.data[2], dim=0))

                                print("grad:", self.gate1 / self.gate1_count)
                                print("grad:", self.gate2 / self.gate2_count)
                                print("grad:", self.gate4 / self.gate4_count)
                                self.gate1 = torch.zeros([1, 3]).cuda(self.cuda)
                                self.gate1_count = 0
                                self.gate2 = torch.zeros([1, 3]).cuda(self.cuda)
                                self.gate2_count = 0
                                self.gate4 = torch.zeros([1, 3]).cuda(self.cuda)
                                self.gate4_count = 0


        print('Finished Training')
        if self.log:
            print("Finished Training", file=open(self.log, "a"))

    def test_overall(self):
        correct0 = 0
        total0 = 0
        correct1 = 0
        total1 = 0
        correct2 = 0
        total2 = 0
        correct3 = 0
        total3 = 0
        with torch.no_grad():
            for data in self.testloader:
                inputs, labels = data
                if self.cuda != None:
                    inputs = inputs.cuda(self.cuda)
                    labels = labels.cuda(self.cuda)

                task_labels = [torch.tensor([t.label_to_task(l) for l in labels]) for t in self.tasks]
                if self.cuda != None:
                    task_labels = [t.cuda(self.cuda) for t in task_labels]


                outputs = self.forward(inputs, test=True)
                predicted = [torch.max(output.data, 1)[1] for output in outputs]

                total0 += task_labels[0].size(0)
                total1 += task_labels[1].size(0)
                total2 += task_labels[2].size(0)
                total3 += task_labels[3].size(0)

                correct0 += (predicted[0] == task_labels[0]).sum().item()
                correct1 += (predicted[1] == task_labels[1]).sum().item()
                correct2 += (predicted[2] == task_labels[2]).sum().item()
                correct3 += (predicted[3] == task_labels[3]).sum().item()

        if self.log:
            print('task_0 accuracy of the network on the 10000 test images: %d %%' % (
                    100 * correct0 / total0), file=open(self.log, "a"))
            print('task_1 accuracy of the network on the 10000 test images: %d %%' % (
                    100 * correct1 / total1), file=open(self.log, "a"))
            print('task_2 accuracy of the network on the 10000 test images: %d %%' % (
                    100 * correct2 / total2), file=open(self.log, "a"))
            print('task_3 accuracy of the network on the 10000 test images: %d %%' % (
                    100 * correct3 / total3), file=open(self.log, "a"))

        print('task_0 accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct0 / total0))
        print('task_1 accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct1 / total1))
        print('task_2 accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct2 / total2))
        print('task_3 accuracy of the network on the 10000 test images: %d %%' % (
               100 * correct3 / total3))


    def log_gate1(self, v):
        self.gate1 += v
        self.gate1_count += 1

    def log_gate2(self, v):
        self.gate2 += v
        self.gate2_count += 1

    def log_gate4(self, v):
        self.gate4 += v
        self.gate4_count += 1

    def add_normal_noise(self, v):
        return v + self.normal.sample()


    def forward(self, img, test=False):  # make number of blocks, tasks variable
        # Task 1
        t1 = self.conv1(img)
        # Task 2, 3
        t23 = self.conv2(img)
        # Task 4
        t4 = self.conv3(img)

        if self.conv_noise and test == False:
            t1.register_hook(self.add_normal_noise)
            t23.register_hook(self.add_normal_noise)
            t4.register_hook(self.add_normal_noise)

        stack = torch.stack([t1.view(-1, self.fc_out), t23.view(-1, self.fc_out), t4.view(-1, self.fc_out)], dim=1)

        # FC & Gate Task 1
        g1 = nn.functional.softmax(self.g_logits[0], dim=0)
        x1 = (stack * g1.view(-1, 1)).sum(dim=1)
        x1 = self.fc2_0(F.relu(self.fc1_0(x1)))

        # FC & Gate Task 2, 3
        g2 = nn.functional.softmax(self.g_logits[1], dim=0)
        x2 = (stack * g2.view(-1, 1)).sum(dim=1)
        loc = F.relu(self.fc1_1(x2))
        x2 = self.fc2_1(loc)
        x3 = self.fc2_2(loc)

        # FC & Gate Task 4
        g4 = nn.functional.softmax(self.g_logits[2], dim=0)
        x4 = (stack * g4.view(-1, 1)).sum(dim=1)
        x4 = self.fc2_3(F.relu(self.fc1_3(x4)))

        if test == False:
            g1.register_hook(self.log_gate1)
            g2.register_hook(self.log_gate2)
            g4.register_hook(self.log_gate4)

        if self.gate_noise and test == False:
            g1.register_hook(self.add_normal_noise)
            g2.register_hook(self.add_normal_noise)
            g4.register_hook(self.add_normal_noise)

        return [self.softmax(x1), self.softmax(x2), self.softmax(x3), self.softmax(x4)]

class LargeNet(GatingNet):
    def __init__(self, tasks, trainloader, testloader, log=None, cuda=None, conv_noise=False, gate_noise=False):
        super(LargeNet, self).__init__(cuda=cuda)
        self.trainloader = trainloader
        self.testloader = testloader
        self.tasks = tasks
        self.log = log
        self.fc_out = 128 * 5 * 5
        self.conv_noise = conv_noise
        self.gate_noise = gate_noise

        self.gate1 = torch.zeros([1, 3])
        self.gate1_count = 0

        self.gate2 = torch.zeros([1, 3])
        self.gate2_count = 0

        self.gate4 = torch.zeros([1, 3])
        self.gate4_count = 0

        if cuda:
            self.gate1 = self.gate1.cuda(cuda)
            self.gate2 = self.gate2.cuda(cuda)
            self.gate4 = self.gate4.cuda(cuda)


        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(3, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(3, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )


        self.fc1_0 = nn.Linear(128 * 5 * 5, 16)
        self.fc2_0 = nn.Linear(16, tasks[0].size)

        self.fc1_1 = nn.Linear(128 * 5 * 5, 16)
        self.fc2_1 = nn.Linear(16, tasks[1].size)
        self.fc2_2 = nn.Linear(16, tasks[2].size)

        self.fc1_3 = nn.Linear(128 * 5 * 5, 16)
        self.fc2_3 = nn.Linear(16, tasks[3].size)

        self.g_logits = nn.Parameter(torch.zeros([3, 3]))
        self.softmax = nn.LogSoftmax(dim=1)
        self.setup()


class SmallNet(GatingNet):
    def __init__(self, tasks, trainloader, testloader, log=None, cuda=None):
        super(SmallNet, self).__init__(cuda=cuda)
        self.trainloader = trainloader
        self.testloader = testloader
        self.tasks = tasks
        self.log = log
        self.fc_out = int(43264/4)


        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(3, 32, 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(3, 32, 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )


        self.fc1_0 = nn.Linear(self.fc_out, 16)
        self.fc2_0 = nn.Linear(16, tasks[0].size)

        self.fc1_1 = nn.Linear(self.fc_out, 16)
        self.fc2_1 = nn.Linear(16, tasks[1].size)
        self.fc2_2 = nn.Linear(16, tasks[2].size)

        self.fc1_3 = nn.Linear(self.fc_out, 16)
        self.fc2_3 = nn.Linear(16, tasks[3].size)

        self.g_logits = nn.Parameter(torch.zeros([3, 3]))
        self.softmax = nn.LogSoftmax(dim=1)
        self.setup()