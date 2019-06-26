import torch
import data_loader
from task import Task
from network import OneLayerBlend, OneLayerSample, OneLayerVI, Individual


local = True
machine = 1

def color_task(l):
    return l[0]

def loc_x_task(l):
    return l[1]

def loc_y_task(l):
    return l[2]

def cifar_task(l):
    return l[3]


cifar_classes = ['aquatic mammals', 'fish', 'flowers', 'food containers', 'fruit and vegetables',
                         'household electrical devices', 'household furniture', 'insects', 'large carnivores',
                         'large man-made outdoor things', 'large natural outdoor scenes',
                         'large omnivores and herbivores', 'medium-sized mammals', 'non-insect invertebrates', 'people',
                         'reptiles', 'small mammals', 'trees', 'vehicles 1', 'vehicles 2']

mnist_classes = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']


def task_assigner(i):
    return lambda l: l[i]


def get_first_task():
    return lambda l: l


cifar_tasks = [Task([i for i in range(5)], get_first_task()) for i in range(20)]
mnist_tasks = [Task([i for i in range(2)], get_first_task()) for i in range(10)]

if local:
    cuda0 = None
else:
    cuda0 = torch.device('cuda:0')


def run_test(config, trials=1, data_amts=[1], path="data/cifar100", dataset="CIFAR100", log="logs/sampling/cifar",
             tasks=None, sample=False, individual=False, VI=False, blending=False):

    if local:
        trials = 1

    for trial in range(trials):
        for data_amt in data_amts:

            crop_size = 32

            if local:
                data_amt = 1/128


            trainloader = data_loader.get_loader(path, dataset=dataset, mode="train", batch_size=32,
                                                 image_size=32, crop_size=crop_size, data_portion=data_amt)

            if local:
                testloader = data_loader.get_loader(path, dataset=dataset, mode="train", batch_size=32,
                                                        image_size=32, crop_size=crop_size, data_portion=1/128)
                validloader = data_loader.get_loader(path, dataset=dataset, mode="valid", batch_size=32,
                                                     image_size=32, crop_size=crop_size, data_portion=1/128)
            else:
                validloader = data_loader.get_loader(path, dataset=dataset, mode="valid", batch_size=32,
                                                     image_size=32, crop_size=crop_size, data_portion=1)
                testloader = data_loader.get_loader(path, dataset=dataset, mode="test", batch_size=32,
                                                    image_size=32, crop_size=crop_size, data_portion=1)

            if log:
                log_str = log+"-"+str(trial)+"-"+str(data_amt)
            else:
                log_str = None

            if sample:
                net = OneLayerSample(tasks, trainloader, validloader, testloader, config=config,
                                     log=log_str, cuda=cuda0)
            elif VI:
                net = OneLayerVI(tasks, trainloader, validloader, testloader, config=config,
                                 log=log_str, cuda=cuda0)
            elif blending:
                net = OneLayerBlend(tasks, trainloader, validloader, testloader, config=config,
                               log=log_str, cuda=cuda0)
            elif individual:
                net = Individual(tasks, trainloader, validloader, testloader, config=config,
                                    log=log_str, cuda=cuda0)

            net = net.to(cuda0)

            if local:
                for i in range(1):
                    net.train_epoch(log_interval=int(len(trainloader) - 1))
                    net.test_overall()
            else:
                for i in range(int(50)):
                    net.train_epoch(log_interval=int(len(trainloader) - 1))
                    net.test_overall()
                    if int(i % 20 * 1/data_amt) == int(19 * 1/data_amt):
                        torch.save(net.state_dict(), log+"-"+str(trial)+"-"+str(data_amt)+"-i-"+str(i)+".pt")


config = {
    "reg": 0.01,
    "reparam": False,
    "discrete": False,
    "in_channels": 3,
    "fc_in": 128,
    "options": 5,
    "blocks": 100,
    "sparse": True,
    "avg_k": False,
    "pick_one": False
}

run_test(config, trials=1, data_amts=[1], path="data/cifar100",
         dataset="CIFAR100", log="logs/sampling/cf-100blk-0.01reg", tasks=cifar_tasks,
         sample=True)

config = {
    "reg": 0.0,
    "reparam": False,
    "discrete": False,
    "in_channels": 3,
    "fc_in": 128,
    "options": 5,
    "blocks": 100,
    "sparse": False
}

run_test(config, trials=1, data_amts=[1], path="data/cifar100",
         dataset="CIFAR100", log="logs/blending/cf-100blk", tasks=cifar_tasks,
         blending=True)