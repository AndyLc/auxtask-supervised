"""
class Trainer():
    def __init__(self, net, trainloader, testloader):


    def execute(self):


"""
"""
Setup
"""

import torch
from noise_position import Position
from task import Task
from gating_net import LargeNet, SmallNet

color_classes = ["red", "green", "blue"]
loc_x_classes = [i for i in range(2, 30)]
loc_y_classes = [i for i in range(2, 30)]
cifar_classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def color_task(l):
    return l[0]

def loc_x_task(l):
    return l[1]

def loc_y_task(l):
    return l[2]

def cifar_task(l):
    return l[3]

color_task = Task(color_classes, color_task)
x_task = Task(loc_x_classes, loc_x_task)
y_task = Task(loc_y_classes, loc_x_task)
cifar_classes = Task(cifar_classes, cifar_task)

tasks = [color_task, x_task, y_task, cifar_classes]

trainset = Position('../cifar_data', train=True, download=True, data_portion=1)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)
testset = Position('../cifar_data', train=False, download=True, data_portion=1)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)


cuda0 = torch.device('cuda:0')

for trial in range(2):
    net = LargeNet(tasks, trainloader, testloader, log="logs/default-" + str(trial), cuda=cuda0)
    net = net.to(cuda0)
    for i in range(300):
        net.train_epoch()
        net.test_overall()
        if i % 20 == 0 and i != 0:
            torch.save(net.state_dict(), "logs/default-t" + str(trial) + "-" + str(i) + ".pt")