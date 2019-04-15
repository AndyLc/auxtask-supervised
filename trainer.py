import torch
import data_loader
from task import Task
from gating_net import LargeNet, RouteNet, OneLayerNet, TwoLayerNet

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

celeb_classes = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']

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

cifar_tasks = [Task([i for i in range(5)], get_first_task()) for i in range(5)]

mnist_tasks = [Task(["not " + mnist_classes[i], mnist_classes[i]], task_assigner(i)) for i in range(10)]
#mnist_tasks = [Task(["not " + mnist_classes[i], mnist_classes[i]], task_assigner(i)) for i in range(1)]

#cuda0 = torch.device('cuda:0')
cuda0 = None


def run_test(trials=1, data_amts=[1], path="data/cifar100", dataset="CIFAR100", log="logs/sampling/cifar",
             tasks=cifar_tasks, sample=False, individual=False, mixed=False, VI=False, naive_sample=False, blocks=3):
    for trial in range(trials):
        for data_amt in data_amts:
            trainloader = data_loader.get_loader(path, dataset=dataset, mode="train", batch_size=32,
                                                 image_size=32, crop_size=32, data_portion=data_amt)
            testloader = data_loader.get_loader(path, dataset=dataset, mode="test", batch_size=32,
                                                image_size=32, crop_size=32, data_portion=data_amt)

            if log:
                log_str = log+"-"+str(trial)+"-"+str(data_amt)
            else:
                log_str = None

            net = OneLayerNet(tasks, trainloader, testloader, 128 * 5 * 5, 3,
                           log=log_str, cuda=cuda0, blocks=blocks)

            net.active_tasks = len(tasks)
            net = net.to(cuda0)

            for i in range(50):
                net.train_epoch(sample=sample, naive_sample=naive_sample, individual=individual, mixture=mixed, VI=VI, log_interval=int(data_amt * 1600))
                net.test_overall(sample=sample, naive_sample=naive_sample, individual=individual, mixture=mixed, VI=VI)
                if i % 20 == 19:
                    torch.save(net.state_dict(), log+"-"+str(trial)+"-"+str(data_amt)+"-i-"+str(i)+".pt")


#run_test(trials=1, data_amts=[1/258], path="data/celeba", dataset="CelebA", log=None, tasks=celebA_tasks, VI=True, blocks=10)
#run_test(trials=1, data_amts=[1/64], path="data/mnist", dataset="MNIST", log=None, tasks=mnist_tasks, individual=True, blocks=10)
#run_test(trials=1, data_amts=[1/8], path="data/celeba", dataset="CelebA", log=None, tasks=celebA_tasks, sample=True, blocks=10)
#run_test(trials=1, data_amts=[1/8], path="data/mnist", dataset="MNIST", log=None, tasks=mnist_tasks, naive_sample=True, blocks=10)
#run_test(trials=1, data_amts=[1/8], path="data/mnist", dataset="MNIST", log=None, tasks=mnist_tasks, blocks=10)


run_test(trials=1, data_amts=[1, 1/8], path="data/celeba", dataset="CelebA", log="logs/sampling/celeb", tasks=celebA_tasks, sample=True, blocks=10)
run_test(trials=1, data_amts=[1, 1/8], path="data/celeba", dataset="CelebA", log="logs/naive_sampling/celeb", tasks=celebA_tasks, naive_sample=True, blocks=10)
run_test(trials=1, data_amts=[1, 1/8], path="data/celeba", dataset="CelebA", log="logs/VI/celeb", tasks=celebA_tasks, VI=True, blocks=10)
run_test(trials=1, data_amts=[1, 1/8], path="data/celeba", dataset="CelebA", log="logs/blending/celeb", tasks=celebA_tasks, blocks=10)
run_test(trials=1, data_amts=[1, 1/8], path="data/celeba", dataset="CelebA", log="logs/individual/celeb", tasks=celebA_tasks, individual=True, blocks=len(celeb_classes))
