"""
Setup
"""
import torch
import torchvision
import torchvision.transforms as transforms
import pandas as pd
from noise_position import Position
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import ast
import os
from torch.multiprocessing import Process


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = Position('./cifar_data', train=True,
                                        download=True)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = Position('./cifar_data', train=False,
                                        download=True)

testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

print("Train Data Count:", len(trainset))
print("Test Data Count:", len(testset))

"""
Tasks Definition
"""

color_classes = ["red", "green", "blue"]
x_axis_classes = [i for i in range(2, 30)]
y_axis_classes = [i for i in range(2, 30)]
cifar_classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def color_task(l):
    return l[0]

def x_axis_task(l):
    return float(l[1])/28

def x_axis_denorm(l):
    return l * 28 + 2

def y_axis_task(l):
    return float(l[2])/28

def y_axis_denorm(l):
    return l * 28 + 2

def cifar_task(l):
    return l[3]

label_to_task0 = color_task
task0_classes = color_classes

label_to_task1 = x_axis_task
task1_to_label = x_axis_denorm

label_to_task2 = y_axis_task
task2_to_label = y_axis_denorm

label_to_task3 = cifar_task
task3_classes = cifar_classes

"""
Model
"""

class Net(nn.Module):

    #nettype looks like [0->4, 0->4, 0->4]
    def __init__(self, netType):
        super(Net, self).__init__()
        self.netType = netType

        #large
        self.conv1_1 = nn.Conv2d(3, 64, 3)
        self.conv1_2 = nn.Conv2d(3, 64, 3)
        self.conv1_3 = nn.Conv2d(3, 64, 3)
        self.conv1_4 = nn.Conv2d(3, 64, 3)

        self.conv2_1 = nn.Conv2d(64, 64, 3)
        self.conv2_2 = nn.Conv2d(64, 64, 3)
        self.conv2_3 = nn.Conv2d(64, 64, 3)
        self.conv2_4 = nn.Conv2d(64, 64, 3)

        self.conv3_1 = nn.Conv2d(64, 128, 3)
        self.conv3_2 = nn.Conv2d(64, 128, 3)
        self.conv3_3 = nn.Conv2d(64, 128, 3)
        self.conv3_4 = nn.Conv2d(64, 128, 3)

        self.conv4_1 = nn.Conv2d(128, 128, 3)
        self.conv4_2 = nn.Conv2d(128, 128, 3)
        self.conv4_3 = nn.Conv2d(128, 128, 3)
        self.conv4_4 = nn.Conv2d(128, 128, 3)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1_0 = nn.Linear(128 * 5 * 5, 16)
        self.fc2_0 = nn.Linear(16, len(task0_classes))

        self.fc1_1 = nn.Linear(128 * 5 * 5, 16)
        self.fc2_1 = nn.Linear(16, 1)

        self.fc1_2 = nn.Linear(128 * 5 * 5, 16)
        self.fc2_2 = nn.Linear(16, 1)

        self.fc1_3 = nn.Linear(128 * 5 * 5, 16)
        self.fc2_3 = nn.Linear(16, len(task3_classes))

        self.softmax = nn.Softmax(dim=1) #LogSoftmax or Softmax?

    def forward(self, x):
        l0 = x
        l1 = F.relu(self.conv1_1(x))
        l2 = self.pool(F.relu(self.conv2_1(l1)))
        l3 = F.relu(self.conv3_1(l2))
        l4 = self.pool(F.relu(self.conv4_1(l3)))

        if self.netType[0] == 0:
            l1_2 = F.relu(self.conv1_2(l0))
            l2_2 = self.pool(F.relu(self.conv2_2(l1_2)))
            l3_2 = F.relu(self.conv3_2(l2_2))
            l4_2 = self.pool(F.relu(self.conv4_2(l3_2)))
        elif self.netType[0] == 1:
            l2_2 = self.pool(F.relu(self.conv2_2(l1)))
            l3_2 = F.relu(self.conv3_2(l2_2))
            l4_2 = self.pool(F.relu(self.conv4_2(l3_2)))
        elif self.netType[0] == 2:
            l3_2 = F.relu(self.conv3_2(l2))
            l4_2 = self.pool(F.relu(self.conv4_2(l3_2)))
        elif self.netType[0] == 3:
            l4_2 = self.pool(F.relu(self.conv4_2(l3)))

        if self.netType[1] == 0:
            l1_3 = F.relu(self.conv1_3(l0))
            l2_3 = self.pool(F.relu(self.conv2_3(l1_3)))
            l3_3 = F.relu(self.conv3_3(l2_3))
            l4_3 = self.pool(F.relu(self.conv4_3(l3_3)))
        elif self.netType[1] == 1:
            l2_3 = self.pool(F.relu(self.conv2_3(l1)))
            l3_3 = F.relu(self.conv3_3(l2_3))
            l4_3 = self.pool(F.relu(self.conv4_3(l3_3)))
        elif self.netType[1] == 2:
            l3_3 = F.relu(self.conv3_3(l2))
            l4_3 = self.pool(F.relu(self.conv4_3(l3_3)))
        elif self.netType[1] == 3:
            l4_3 = self.pool(F.relu(self.conv4_3(l3)))

        if self.netType[2] == 0:
            l1_4 = F.relu(self.conv1_4(l0))
            l2_4 = self.pool(F.relu(self.conv2_4(l1_4)))
            l3_4 = F.relu(self.conv3_4(l2_4))
            l4_4 = self.pool(F.relu(self.conv4_4(l3_4)))
        elif self.netType[2] == 1:
            l2_4 = self.pool(F.relu(self.conv2_4(l1)))
            l3_4 = F.relu(self.conv3_4(l2_4))
            l4_4 = self.pool(F.relu(self.conv4_4(l3_4)))
        elif self.netType[2] == 2:
            l3_4 = F.relu(self.conv3_4(l2))
            l4_4 = self.pool(F.relu(self.conv4_4(l3_4)))
        elif self.netType[2] == 3:
            l4_4 = self.pool(F.relu(self.conv4_4(l3)))

        x1 = l4.view(-1, 128 * 5 * 5)
        x1 = self.fc2_0(F.relu(self.fc1_0(x1)))

        if self.netType[0] == 4:
            x2 = l4.view(-1, 128 * 5 * 5)
            x2 = self.fc2_1(F.relu(self.fc1_1(x2)))
        else:
            x2 = l4_2.view(-1, 128 * 5 * 5)
            x2 = self.fc2_1(F.relu(self.fc1_1(x2)))

        if self.netType[1] == 4:
            x3 = l4.view(-1, 128 * 5 * 5)
            x3 = self.fc2_2(F.relu(self.fc1_2(x3)))
        else:
            x3 = l4_3.view(-1, 128 * 5 * 5)
            x3 = self.fc2_2(F.relu(self.fc1_2(x3)))

        if self.netType[2] == 4:
            x4 = l4.view(-1, 128 * 5 * 5)
            x4 = self.fc2_3(F.relu(self.fc1_3(x4)))
        else:
            x4 = l4_4.view(-1, 128 * 5 * 5)
            x4 = self.fc2_3(F.relu(self.fc1_3(x4)))

        return self.softmax(x1), x2, x3, self.softmax(x4)

def train(net, optimizer, c_crit, r_crit, cud=None, epochs=1):
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            if cud != None:
                inputs = inputs.cuda(cud, async=True)
                labels = labels.cuda(cud, async=True)

            labels1 = torch.tensor([label_to_task0(l) for l in labels])
            labels2 = torch.tensor([label_to_task1(l) for l in labels], dtype=torch.float)
            labels3 = torch.tensor([label_to_task2(l) for l in labels], dtype=torch.float)
            labels4 = torch.tensor([label_to_task3(l) for l in labels])

            if cud != None:
                labels1.cuda(cud, async=True)
                labels2.cuda(cud, async=True)
                labels3.cuda(cud, async=True)
                labels4.cuda(cud, async=True)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs1, outputs2, outputs3, outputs4 = net(inputs)
            loss1 = c_crit(outputs1, labels1)
            loss2 = r_crit(outputs2, labels2)
            loss3 = r_crit(outputs3, labels3)
            loss4 = c_crit(outputs4, labels4)
            loss = (loss1 + loss2 + loss3 + loss4) / 4
            loss.backward()
            optimizer.step()

            # print statistics
            with torch.no_grad():
                running_loss += loss.item()
                if i % 1000 == 999:    # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 1000))
                    running_loss = 0.0

    print('Finished Training')


def test_overall(net, performance=None):
    correct0 = 0
    total0 = 0
    error1 = 0
    total1 = 0
    error2 = 0
    total2 = 0
    correct3 = 0
    total3 = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            labels0 = torch.tensor([label_to_task0(l) for l in labels])
            labels1 = torch.tensor([label_to_task1(l) for l in labels], dtype=torch.float)
            labels2 = torch.tensor([label_to_task2(l) for l in labels], dtype=torch.float)
            labels3 = torch.tensor([label_to_task3(l) for l in labels])
            outputs0, outputs1, outputs2, outputs3 = net(images)
            _, predicted0 = torch.max(outputs0.data, 1)
            _, predicted3 = torch.max(outputs3.data, 1)
            total0 += labels0.size(0)
            total1 += labels1.size(0)
            total2 += labels2.size(0)
            total3 += labels3.size(0)
            correct0 += (predicted0 == labels0).sum().item()
            error1 += ((outputs1 - labels1) * (outputs1 - labels1)).sum().item()
            error2 += ((outputs2 - labels2) * (outputs2 - labels2)).sum().item()
            correct3 += (predicted3 == labels3).sum().item()


    print('task_0 accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct0 / total0))

    print('task_1 error of the network on the 10000 test images: %d %%' % (
        100 * error1 / (total1*4)))

    print('task_2 error of the network on the 10000 test images: %d %%' % (
        100 * error2 / (total2*4)))

    print('task_3 accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct3 / total3))

    if performance != None:
        performance["task_0"].append(100 * correct0 / total0)
        performance["task_1"].append(100 * error1 / total1)
        performance["task_2"].append(100 * error2 / total2)
        performance["task_3"].append(100 * correct3 / total3)


def get_performance_data(net, c_crit, r_crit, optimizer, epochs=1, cud=None):
    performance = {'task_0':[], 'task_1':[], 'task_2':[], 'task_3':[]}

    for e in range(epochs):
        train(net, optimizer, c_crit, r_crit, epochs=1, cud=cud)
        test_overall(net, performance)
        #test_individual(net, performance)

    return performance

def train_job(configs, p_num, cud=None):
    np.random.seed(p_num)
    torch.manual_seed(p_num)
    for config in configs:
        for i in range(1, 6):
            performances = []
            for _ in range(1):
                net = Net(config)
                if cud != None:
                    net.cuda(cud)

                criterion = nn.CrossEntropyLoss()#.cuda(cud)

                optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
                performance1 = get_performance_data(net, criterion, optimizer, epochs=15, cud=cud)

                optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=0.9, weight_decay=5e-4)
                performance2 = get_performance_data(net, criterion, optimizer, epochs=10, cud=cud)

                optimizer = optim.SGD(net.parameters(), lr=0.0025, momentum=0.9, weight_decay=5e-4)
                performance3 = get_performance_data(net, criterion, optimizer, epochs=5, cud=cud)

                optimizer = optim.SGD(net.parameters(), lr=0.00125, momentum=0.9, weight_decay=5e-4)
                performance4 = get_performance_data(net, criterion, optimizer, epochs=5, cud=cud)

                performance = {}
                for p in performance1.keys():
                    performance[p] = performance1[p]

                performances.append(performance)

                df = pd.DataFrame(data=performances)
                df.to_csv("mixed-tasks/" + ''.join([str(x) for x in config]) + "_iter" + str(i) + '.csv')

            print("Trained:", i)


"""
Automated Testing
"""

if __name__ == '__main__':

    #cuda0 = torch.device('cuda:0')
    #cuda1 = torch.device('cuda:1')
    #cuda2 = torch.device('cuda:2')
    #cuda3 = torch.device('cuda:3')

    procs = []

    print("Starting process 0")
    proc = Process(target=train_job, args=([[4, 4, 1], [2, 2, 4], [2, 2, 1], [2, 2, 0], [3, 3, 4], [3, 3, 2], [3, 3, 0]], 0, None))
    procs.append(proc)
    proc.start()

    print("Starting process 1")
    proc = Process(target=train_job, args=([[4, 4, 3], [2, 2, 3], [2, 2, 2], [0, 0, 0], [3, 3, 4], [3, 3, 3]], 1, None))
    procs.append(proc)
    proc.start()


    for proc in procs:
        proc.join()
