import torch
import data_loader
from task import Task
from network import OneLayerBlend, OneLayerSample, OneLayerVI, Individual
#from gating_net import LargeNet, RouteNet, OneLayerNet, SampledOneLayerNet


local = True



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
#celeb_classes = ['Bushy_Eyebrows', 'No_Beard', 'Brown_Hair', 'Blond_Hair', 'Black_Hair', 'Eyeglasses']

#celeb_classes = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']
celeb_classes = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive']

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

celebA_tasks = [Task(["Not" + celeb_classes[i], celeb_classes[i]], task_assigner(i)) for i in range(len(celeb_classes))]
#celebA_tasks[0].scale = 3

cifar_tasks = [Task([i for i in range(5)], get_first_task()) for i in range(20)]
mnist_tasks = [Task([i for i in range(7)], get_first_task()) for i in range(2)]
mitstates_tasks = [Task([i for i in range(5)], get_first_task()) for i in range(228)]

if local:
    cuda0 = None
else:
    cuda0 = torch.device('cuda:0')


def run_test(config, trials=1, data_amts=[1], path="data/cifar100", dataset="CIFAR100", log="logs/sampling/cifar",
             tasks=None, sample=False, individual=False, VI=False, blending=False):
    for trial in range(trials):
        for data_amt in data_amts:

            if dataset == "CelebA":
                crop_size = 128
            else:
                crop_size = 32

            trainloader = data_loader.get_loader(path, dataset=dataset, mode="train", batch_size=32,
                                                 image_size=32, crop_size=crop_size, data_portion=data_amt)

            if local:
                testloader = data_loader.get_loader(path, dataset=dataset, mode="train", batch_size=32,
                                                        image_size=32, crop_size=crop_size, data_portion=data_amt)
            else:
                testloader = data_loader.get_loader(path, dataset=dataset, mode="test", batch_size=32,
                                                    image_size=32, crop_size=crop_size, data_portion=data_amt)

            if log:
                log_str = log+"-"+str(trial)+"-"+str(data_amt)
            else:
                log_str = None

            if sample:
                net = OneLayerSample(tasks, trainloader, testloader, config=config,
                                     log=log_str, cuda=cuda0)
            elif VI:
                net = OneLayerVI(tasks, trainloader, testloader, config=config,
                                 log=log_str, cuda=cuda0)
            elif blending:
                net = OneLayerBlend(tasks, trainloader, testloader, config=config,
                               log=log_str, cuda=cuda0)
            elif individual:
                net = Individual(tasks, trainloader, testloader, config=config,
                                    log=log_str, cuda=cuda0)

            net = net.to(cuda0)

            if local:
                for i in range(10):
                    net.train_epoch(log_interval=int(len(trainloader) - 1))
                    #net.test_overall()
            else:
                for i in range(int(100)):
                    net.train_epoch(log_interval=int(len(trainloader) - 1))
                    net.test_overall()
                    if int(i % 20 * 1/data_amt) == int(19 * 1/data_amt):
                        torch.save(net.state_dict(), log+"-"+str(trial)+"-"+str(data_amt)+"-i-"+str(i)+".pt")


"""

CIFAR TESTS

"""

"""

config = {
    "reg": 0.01,
    "reparam": True,
    "discrete": True,
    "in_channels": 3,
    "fc_in": 128 * 5 * 5,
    "options": 5,
    "blocks": 10
}

run_test(config, trials=1, data_amts=[1/8], path="data/cifar100", dataset="CIFAR100", log=None, tasks=cifar_tasks,
         VI=True)
if local:
    run_test(config, trials=1, data_amts=[1/128], path="data/cifar100", dataset="CIFAR100", log=None, tasks=cifar_tasks, VI=True)
    run_test(config, trials=1, data_amts=[1/128], path="data/cifar100", dataset="CIFAR100", log=None, tasks=cifar_tasks, sample=True)
else:
    run_test(config, trials=1, data_amts=[1, 1/2, 1/4, 1/8, 1/16, 1/32, 1/64], path="data/cifar100", dataset="CIFAR100", log="logs/VI/cifar_rep_disc", tasks=cifar_tasks, VI=True)
    run_test(config, trials=1, data_amts=[1, 1/2, 1/4, 1/8, 1/16, 1/32, 1/64], path="data/cifar100", dataset="CIFAR100", log="logs/sampling/cifar_rep_disc", tasks=cifar_tasks, sample=True)

config = {
    "reg": 0.01,
    "reparam": True,
    "discrete": False,
    "in_channels": 3,
    "fc_in": 128 * 5 * 5,
    "options": 5,
    "blocks": 10
}

if local:
    run_test(config, trials=1, data_amts=[1/128], path="data/cifar100", dataset="CIFAR100", log=None, tasks=cifar_tasks, VI=True)
    run_test(config, trials=1, data_amts=[1/128], path="data/cifar100", dataset="CIFAR100", log=None, tasks=cifar_tasks, sample=True)
else:
    run_test(config, trials=1, data_amts=[1, 1/2, 1/4, 1/8, 1/16, 1/32, 1/64], path="data/cifar100", dataset="CIFAR100", log="logs/VI/cifar_rep_nodisc", tasks=cifar_tasks, VI=True)
    run_test(config, trials=1, data_amts=[1, 1/2, 1/4, 1/8, 1/16, 1/32, 1/64], path="data/cifar100", dataset="CIFAR100", log="logs/sampling/cifar_rep_nodisc", tasks=cifar_tasks, sample=True)

config = {
    "reg": 0.01,
    "reparam": False,
    "discrete": False,
    "in_channels": 3,
    "fc_in": 128 * 5 * 5,
    "options": 5,
    "blocks": 10
}

if local:
    run_test(config, trials=1, data_amts=[1/128], path="data/cifar100", dataset="CIFAR100", log=None, tasks=cifar_tasks, VI=True)
    run_test(config, trials=1, data_amts=[1/128], path="data/cifar100", dataset="CIFAR100", log=None, tasks=cifar_tasks, sample=True)
else:
    run_test(config, trials=1, data_amts=[1, 1/2, 1/4, 1/8, 1/16, 1/32, 1/64], path="data/cifar100", dataset="CIFAR100", log="logs/VI/cifar_score_disc", tasks=cifar_tasks, VI=True)
    run_test(config, trials=1, data_amts=[1, 1/2, 1/4, 1/8, 1/16, 1/32, 1/64], path="data/cifar100", dataset="CIFAR100", log="logs/sampling/cifar_score_disc", tasks=cifar_tasks, sample=True)

if local:
    run_test(config, trials=1, data_amts=[1/128], path="data/cifar100", dataset="CIFAR100", log=None, tasks=cifar_tasks, blending=True)
else:
    run_test(config, trials=1, data_amts=[1, 1/2, 1/4, 1/8, 1/16, 1/32, 1/64], path="data/cifar100", dataset="CIFAR100", log="logs/blending/cifar", tasks=cifar_tasks, blending=True)

config = {
    "reg": 0.01,
    "reparam": False,
    "discrete": False,
    "in_channels": 3,
    "fc_in": 128 * 5 * 5,
    "options": 5,
    "blocks": 1
}

if local:
    run_test(config, trials=1, data_amts=[1/128], path="data/cifar100", dataset="CIFAR100", log=None, tasks=cifar_tasks, blending=True)
else:
    run_test(config, trials=1, data_amts=[1, 1/2, 1/4, 1/8, 1/16, 1/32, 1/64], path="data/cifar100", dataset="CIFAR100", log="logs/shared/cifar", tasks=cifar_tasks, blending=True)


config = {
    "reg": 0.01,
    "reparam": False,
    "discrete": False,
    "in_channels": 3,
    "fc_in": 128 * 5 * 5,
    "options": 5,
    "blocks": 20
}

if local:
    run_test(config, trials=1, data_amts=[1/128], path="data/cifar100", dataset="CIFAR100", log=None, tasks=cifar_tasks, individual=True)
else:
    run_test(config, trials=1, data_amts=[1, 1/2, 1/4, 1/8, 1/16, 1/32, 1/64], path="data/cifar100", dataset="CIFAR100", log="logs/individual/cifar", tasks=cifar_tasks, individual=True)

"""

"""

MNIST TESTS

"""

"""

config = {
    "reg": 0.01,
    "reparam": True,
    "discrete": True,
    "in_channels": 1,
    "fc_in": 128 * 5 * 5,
    "options": 7,
    "blocks": 10
}

if local:
    run_test(config, trials=1, data_amts=[1/128], path="data/mnist", dataset="MNIST", log=None, tasks=mnist_tasks, VI=True)
    run_test(config, trials=1, data_amts=[1/128], path="data/mnist", dataset="MNIST", log=None, tasks=mnist_tasks, sample=True)
else:
    run_test(config, trials=1, data_amts=[1, 1/2, 1/4, 1/8, 1/16, 1/32, 1/64], path="data/mnist", dataset="MNIST", log="logs/VI/mnist_rep_disc", tasks=mnist_tasks, VI=True)
    run_test(config, trials=1, data_amts=[1, 1/2, 1/4, 1/8, 1/16, 1/32, 1/64], path="data/mnist", dataset="MNIST", log="logs/sampling/mnist_rep_disc", tasks=mnist_tasks, sample=True)

config = {
    "reg": 0.01,
    "reparam": True,
    "discrete": False,
    "in_channels": 1,
    "fc_in": 128 * 5 * 5,
    "options": 7,
    "blocks": 10
}

if local:
    run_test(config, trials=1, data_amts=[1/128], path="data/mnist", dataset="MNIST", log=None, tasks=mnist_tasks, VI=True)
    run_test(config, trials=1, data_amts=[1/128], path="data/mnist", dataset="MNIST", log=None, tasks=mnist_tasks, sample=True)
else:
    run_test(config, trials=1, data_amts=[1, 1/2, 1/4, 1/8, 1/16, 1/32, 1/64], path="data/mnist", dataset="MNIST", log="logs/VI/mnist_rep_nodisc", tasks=mnist_tasks, VI=True)
    run_test(config, trials=1, data_amts=[1, 1/2, 1/4, 1/8, 1/16, 1/32, 1/64], path="data/mnist", dataset="MNIST", log="logs/sampling/mnist_rep_nodisc", tasks=mnist_tasks, sample=True)

config = {
    "reg": 0.01,
    "reparam": False,
    "discrete": False,
    "in_channels": 1,
    "fc_in": 128 * 5 * 5,
    "options": 7,
    "blocks": 10
}

if local:
    run_test(config, trials=1, data_amts=[1/128], path="data/mnist", dataset="MNIST", log=None, tasks=mnist_tasks, VI=True)
    run_test(config, trials=1, data_amts=[1/128], path="data/mnist", dataset="MNIST", log=None, tasks=mnist_tasks, sample=True)
else:
    run_test(config, trials=1, data_amts=[1, 1/2, 1/4, 1/8, 1/16, 1/32, 1/64], path="data/mnist", dataset="MNIST", log="logs/VI/mnist_score_disc", tasks=mnist_tasks, VI=True)
    run_test(config, trials=1, data_amts=[1, 1/2, 1/4, 1/8, 1/16, 1/32, 1/64], path="data/mnist", dataset="MNIST", log="logs/sampling/mnist_score_disc", tasks=mnist_tasks, sample=True)

if local:
    run_test(config, trials=1, data_amts=[1/128], path="data/mnist", dataset="MNIST", log=None, tasks=mnist_tasks, blending=True)
else:
    run_test(config, trials=1, data_amts=[1, 1/2, 1/4, 1/8, 1/16, 1/32, 1/64], path="data/mnist", dataset="MNIST", log="logs/blending/mnist", tasks=mnist_tasks, blending=True)

config = {
    "reg": 0.01,
    "reparam": False,
    "discrete": False,
    "in_channels": 1,
    "fc_in": 128 * 5 * 5,
    "options": 7,
    "blocks": 1
}

if local:
    run_test(config, trials=1, data_amts=[1/128], path="data/mnist", dataset="MNIST", log=None, tasks=mnist_tasks, blending=True)
else:
    run_test(config, trials=1, data_amts=[1, 1/2, 1/4, 1/8, 1/16, 1/32, 1/64], path="data/mnist", dataset="MNIST", log="logs/shared/mnist", tasks=mnist_tasks, blending=True)


config = {
    "reg": 0.01,
    "reparam": False,
    "discrete": False,
    "in_channels": 1,
    "fc_in": 128 * 5 * 5,
    "options": 7,
    "blocks": 2
}

if local:
    run_test(config, trials=1, data_amts=[1/128], path="data/mnist", dataset="MNIST", log=None, tasks=mnist_tasks, individual=True)
else:
    run_test(config, trials=1, data_amts=[1, 1/2, 1/4, 1/8, 1/16, 1/32, 1/64], path="data/mnist", dataset="MNIST", log="logs/individual/mnist", tasks=mnist_tasks, individual=True)

"""
"""

MITStates

"""

config = {
    "reg": 0.01,
    "reparam": True,
    "discrete": True,
    "in_channels": 3,
    "fc_in": 128 * 5 * 5,
    "options": 5,
    "blocks": 10
}

if local:
    run_test(config, trials=1, data_amts=[1/128], path="data/mitstates", dataset="MITStates", log=None, tasks=mitstates_tasks, VI=True)
    run_test(config, trials=1, data_amts=[1/128], path="data/mitstates", dataset="MITStates", log=None, tasks=mitstates_tasks, sample=True)
else:
    run_test(config, trials=1, data_amts=[1, 1/2, 1/4, 1/8, 1/16, 1/32, 1/64], path="data/mitstates", dataset="MITStates", log="logs/VI/mitstates_rep_disc", tasks=mitstates_tasks, VI=True)
    run_test(config, trials=1, data_amts=[1, 1/2, 1/4, 1/8, 1/16, 1/32, 1/64], path="data/mitstates", dataset="MITStates", log="logs/sampling/mitstates_rep_disc", tasks=mitstates_tasks, sample=True)

config = {
    "reg": 0.01,
    "reparam": True,
    "discrete": False,
    "in_channels": 3,
    "fc_in": 128 * 5 * 5,
    "options": 5,
    "blocks": 10
}

if local:
    run_test(config, trials=1, data_amts=[1/128], path="data/mitstates", dataset="MITStates", log=None, tasks=mitstates_tasks, VI=True)
    run_test(config, trials=1, data_amts=[1/128], path="data/mitstates", dataset="MITStates", log=None, tasks=mitstates_tasks, sample=True)
else:
    run_test(config, trials=1, data_amts=[1, 1/2, 1/4, 1/8, 1/16, 1/32, 1/64], path="data/mitstates", dataset="MITStates", log="logs/VI/mitstates_rep_nodisc", tasks=mitstates_tasks, VI=True)
    run_test(config, trials=1, data_amts=[1, 1/2, 1/4, 1/8, 1/16, 1/32, 1/64], path="data/mitstates", dataset="MITStates", log="logs/sampling/mitstates_rep_nodisc", tasks=mitstates_tasks, sample=True)

config = {
    "reg": 0.01,
    "reparam": False,
    "discrete": False,
    "in_channels": 3,
    "fc_in": 128 * 5 * 5,
    "options": 5,
    "blocks": 10
}

if local:
    run_test(config, trials=1, data_amts=[1/128], path="data/mitstates", dataset="MITStates", log=None, tasks=mitstates_tasks, VI=True)
    run_test(config, trials=1, data_amts=[1/128], path="data/mitstates", dataset="MITStates", log=None, tasks=mitstates_tasks, sample=True)
else:
    run_test(config, trials=1, data_amts=[1, 1/2, 1/4, 1/8, 1/16, 1/32, 1/64], path="data/mitstates", dataset="MITStates", log="logs/VI/mitstates_score_disc", tasks=mitstates_tasks, VI=True)
    run_test(config, trials=1, data_amts=[1, 1/2, 1/4, 1/8, 1/16, 1/32, 1/64], path="data/mitstates", dataset="MITStates", log="logs/sampling/mitstates_score_disc", tasks=mitstates_tasks, sample=True)

if local:
    run_test(config, trials=1, data_amts=[1/128], path="data/mitstates", dataset="MITStates", log=None, tasks=mitstates_tasks, blending=True)
else:
    run_test(config, trials=1, data_amts=[1, 1/2, 1/4, 1/8, 1/16, 1/32, 1/64], path="data/mitstates", dataset="MITStates", log="logs/blending/mitstates", tasks=mitstates_tasks, blending=True)

config = {
    "reg": 0.01,
    "reparam": False,
    "discrete": False,
    "in_channels": 3,
    "fc_in": 128 * 5 * 5,
    "options": 5,
    "blocks": 1
}

if local:
    run_test(config, trials=1, data_amts=[1/128], path="data/mitstates", dataset="MITStates", log=None, tasks=mitstates_tasks, blending=True)
else:
    run_test(config, trials=1, data_amts=[1, 1/2, 1/4, 1/8, 1/16, 1/32, 1/64], path="data/mitstates", dataset="MITStates", log="logs/shared/mitstates", tasks=mitstates_tasks, blending=True)
