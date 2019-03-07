import torch
from noise_position import Position
from task import Task
from gating_net import LargeNet

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
cuda = cuda0

def test_robust_predictions(path):
    #load model parameters

    #add random noise to parameters, ~10 times for 10 different noise levels.
    for noise in [0, 0.01, 0.02, 0.05, 0.1]:
        with torch.no_grad():
            for trial in range(5):
                net = LargeNet(tasks, trainloader, testloader)
                net.load_state_dict(torch.load(path))

                for param in net.parameters():
                    param.add_(torch.randn(param.size()) * noise)

                if cuda:
                    net = net.to(cuda)
                #calculate loss / performance and print
                running_loss_total = 0.0
                for i, data in enumerate(trainloader):
                    inputs, labels = data
                    if cuda != None:
                        inputs = inputs.cuda(cuda)
                        labels = labels.cuda(cuda)

                    task_labels = [torch.tensor([t.label_to_task(l) for l in labels]) for t in net.tasks]

                    if cuda != None:
                        task_labels = [t.cuda(cuda) for t in task_labels]

                    # zero the parameter gradients
                    net.optimizer.zero_grad()
                    losses = [net.criterion(out, label) for out, label in zip(net.forward(inputs, test=True), task_labels)]
                    loss = (losses[0] + losses[1] / 3 + losses[2] / 3 + losses[3] * 2) / 4

                    running_loss_total += loss.item()
                    if i % 5000 == 4999:
                        break;

                print('noise: %.3f, trial: %d, loss: %.3f' % (noise, trial, running_loss_total / 5000))
                #net.test_overall()


test_robust_predictions("default/default-t1-280.pt")
print(" ")
test_robust_predictions("default/default-t1-160.pt")