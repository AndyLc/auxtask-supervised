import torch
import data_loader
from task import Task
from gating_net import LargeNet, LargerNet, AsymmNet

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

color_task = Task(color_classes, color_task, scale=1)
x_task = Task(loc_x_classes, loc_x_task, scale=1/3)
#y_task = Task(loc_y_classes, loc_y_task)
cifar_classes = Task(cifar_classes, cifar_task, scale=2)

tasks = [color_task, x_task, cifar_classes]

#['Bushy_Eyebrows', 'No_Beard', 'Brown_Hair', 'Blond_Hair', 'Black_Hair', 'Eyeglasses']
celeb_classes = ['Bushy_Eyebrows', 'No_Beard', 'Brown_Hair', 'Blond_Hair', 'Black_Hair', 'Eyeglasses']

def task_assigner(i):
    return lambda l: l[i]

celebA_tasks = [Task(["Not" + celeb_classes[i], celeb_classes[i]], task_assigner(i)) for i in range(len(celeb_classes))]
#celebA_tasks[0].scale = 3

#trainloader = data_loader.get_loader(dataset="Superimposed", mode="train", batch_size=32)
#testloader = data_loader.get_loader(dataset="Superimposed", mode="test", batch_size=32)

cuda0 = torch.device('cuda:0')
cuda0 = None


for trial in range(1):
    for data_amt in [1/16]:
        trainloader = data_loader.get_loader("data/celeba", dataset="CelebA", mode="train", batch_size=32,
                                             image_size=32, data_portion=data_amt)
        testloader = data_loader.get_loader("data/celeba", dataset="CelebA", mode="test", batch_size=32, image_size=32, data_portion=data_amt)
        #net = LargeNet(celebA_tasks, trainloader, testloader, 128 * 5 * 5, log="logs/sampling/1-" + str(data_amt), cuda=cuda0)
        net = LargeNet(celebA_tasks, trainloader, testloader, 128 * 5 * 5, log=None,
                       cuda=None)
        #net = net.to(cuda0)
        for i in range(20):
            net.train_epoch(sample=True, individual=False, log_interval=int(data_amt*2500))
            net.test_overall(sample=True, individual=False)
            #if i % 20 == 0:
            #    torch.save(net.state_dict(), "logs/sampling/1-" + str(data_amt) + "-i-" + str(i) + ".pt")

"""

for trial in range(1):
    for data_amt in [1, 1/2, 1/4, 1/8, 1/16]:
        trainloader = data_loader.get_loader("data/celeba", dataset="CelebA", mode="train", batch_size=32,
                                             image_size=32, data_portion=data_amt)
        testloader = data_loader.get_loader("data/celeba", dataset="CelebA", mode="test", batch_size=32, image_size=32, data_portion=data_amt)
        net = LargeNet(celebA_tasks, trainloader, testloader, 128 * 5 * 5, log="logs/blending/1-" + str(data_amt), cuda=cuda0)
        net = net.to(cuda0)
        for i in range(20):
            net.train_epoch(sample=False, individual=False)
            net.test_overall(sample=False, individual=False)
            if i % 20 == 0:
                torch.save(net.state_dict(), "logs/blending/1-" + str(data_amt) + "-i-" + str(i) + ".pt")

for trial in range(1):
    for data_amt in [1, 1/2, 1/4, 1/8, 1/16]:
        trainloader = data_loader.get_loader("data/celeba", dataset="CelebA", mode="train", batch_size=32,
                                             image_size=32, data_portion=data_amt)
        testloader = data_loader.get_loader("data/celeba", dataset="CelebA", mode="test", batch_size=32, image_size=32, data_portion=data_amt)
        net = LargeNet(celebA_tasks, trainloader, testloader, 128 * 5 * 5, log="logs/individual/1-" + str(data_amt), cuda=cuda0)
        net = net.to(cuda0)
        for i in range(20):
            net.train_epoch(sample=False, individual=True)
            net.test_overall(sample=False, individual=True)
            if i % 20 == 0:
                torch.save(net.state_dict(), "logs/individual/1-" + str(data_amt) + "-i-" + str(i) + ".pt")

"""

"""
for trial in range(1):
    for data_amt in [1]:
        trainloader = data_loader.get_loader("data/celeba", dataset="CelebA", mode="train", batch_size=32,
                                             image_size=32, data_portion=data_amt)
        testloader = data_loader.get_loader("data/celeba", dataset="CelebA", mode="test", batch_size=32, image_size=32, data_portion=data_amt)
        net = LargeNet(celebA_tasks, trainloader, testloader, 128 * 5 * 5, log=None, cuda=None)
        #net = net.to(cuda0)
        for i in range(1):
            net.train_epoch(sample=False, individual=False, mixture=True)
            net.test_overall(sample=False, individual=False, mixture=True)
            if i % 20 == 0:
                torch.save(net.state_dict(), "logs/mixed/celeb-i-" + str(i) + ".pt")
"""