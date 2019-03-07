"""
Setup
"""
import torch
import pandas as pd
from noise_position import Position
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os


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

label_to_task0 = color_task
task0_classes = color_classes

label_to_task1 = loc_x_task
task1_classes = loc_x_classes

label_to_task2 = loc_y_task
task2_classes = loc_y_classes

label_to_task3 = cifar_task
task3_classes = cifar_classes


"""
Model
"""

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, 3)
        self.conv1_2 = nn.Conv2d(3, 64, 3)
        self.conv1_3 = nn.Conv2d(3, 64, 3)

        self.conv2_1 = nn.Conv2d(64, 64, 3)
        self.conv2_2 = nn.Conv2d(64, 64, 3)
        self.conv2_3 = nn.Conv2d(64, 64, 3)

        self.conv3_1 = nn.Conv2d(64, 128, 3)
        self.conv3_2 = nn.Conv2d(64, 128, 3)
        self.conv3_3 = nn.Conv2d(64, 128, 3)

        self.conv4_1 = nn.Conv2d(128, 128, 3)
        self.conv4_2 = nn.Conv2d(128, 128, 3)
        self.conv4_3 = nn.Conv2d(128, 128, 3)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1_0 = nn.Linear(128 * 5 * 5, 16)
        self.fc2_0 = nn.Linear(16, len(task0_classes))

        self.fc1_1 = nn.Linear(128 * 5 * 5, 16)
        self.fc2_1 = nn.Linear(16, len(task1_classes))
        self.fc2_2 = nn.Linear(16, len(task2_classes))

        self.fc1_3 = nn.Linear(128 * 5 * 5, 16)
        self.fc2_3 = nn.Linear(16, len(task3_classes))

        self.g_logits = nn.Parameter(torch.zeros([3, 3]))

        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, img): # make number of blocks, tasks variable
        #Task 1
        l0 = img
        l1 = F.relu(self.conv1_1(l0))
        l2 = self.pool(F.relu(self.conv2_1(l1)))
        l3 = F.relu(self.conv3_1(l2))
        l4 = self.pool(F.relu(self.conv4_1(l3)))

        #Task 2, 3
        l1_1 = F.relu(self.conv1_2(l0))
        l2_1 = self.pool(F.relu(self.conv2_2(l1_1)))
        l3_1 = F.relu(self.conv3_2(l2_1))
        l4_1 = self.pool(F.relu(self.conv4_2(l3_1)))

        #Task 4
        l1_2 = F.relu(self.conv1_3(l0))
        l2_2 = self.pool(F.relu(self.conv2_3(l1_2)))
        l3_2 = F.relu(self.conv3_3(l2_2))
        l4_2 = self.pool(F.relu(self.conv4_3(l3_2)))

        stack = torch.stack([l4.view(-1, 128 * 5 * 5), l4_1.view(-1, 128 * 5 * 5), l4_2.view(-1, 128 * 5 * 5)], dim=1)

        #FC & Gate Task 1
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

        return self.softmax(x1), self.softmax(x2), self.softmax(x3), self.softmax(x4)



class Executor():

    def train(self, net, optimizer, c_crit, log=None, cud=None, epochs=1, log_interval=1000):
        print("Starting Training")
        for epoch in range(epochs):
            running_loss_total = 0.0
            running_loss_1 = 0.0
            running_loss_2 = 0.0
            running_loss_3 = 0.0
            running_loss_4 = 0.0
            for i, data in enumerate(self.trainloader, 0):
                # get the inputs
                inputs, labels = data
                if cud != None:
                    inputs = inputs.cuda(cud)
                    labels = labels.cuda(cud)

                labels1 = torch.tensor([label_to_task0(l) for l in labels])
                labels2 = torch.tensor([label_to_task1(l) for l in labels])
                labels3 = torch.tensor([label_to_task2(l) for l in labels])
                labels4 = torch.tensor([label_to_task3(l) for l in labels])

                if cud != None:
                    labels1 = labels1.cuda(cud)
                    labels2 = labels2.cuda(cud)
                    labels3 = labels3.cuda(cud)
                    labels4 = labels4.cuda(cud)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs1, outputs2, outputs3, outputs4 = net(inputs)
                loss1 = c_crit(outputs1, labels1)
                loss2 = c_crit(outputs2, labels2)
                loss3 = c_crit(outputs3, labels3)
                loss4 = c_crit(outputs4, labels4)


                #Entropy regularization
                """
                for name, param in net.named_parameters():
                    if param.requires_grad:
                        if name == "g_logits":
                            print(name, param.data)
                            distr0 = nn.functional.softmax(param.data[0], dim=0)
                            log0 = torch.mul(torch.log2(distr0), -1)
                            print(nn.functional.softmax(param.data[1], dim=0))
                            distr1 = nn.functional.softmax(param.data[0], dim=0)
                            log1 = torch.mul(torch.log2(distr1), -1)
                            print(nn.functional.softmax(param.data[2], dim=0))
                            distr2 = nn.functional.softmax(param.data[0], dim=0)
                            log2 = torch.mul(torch.log2(distr2), -1)
                            entropy_loss = torch.sum(distr0 * log0) + torch.sum(distr1 * log1) + torch.sum(distr2 * log2)
                """

                loss = (loss1 + loss2 / 3 + loss3 / 3 + loss4 * 2) / 4 #+ 0.01*entropy_loss
                loss.backward()
                optimizer.step()

                # print statistics
                with torch.no_grad():
                    running_loss_total += loss.item()
                    running_loss_1 += loss1.item()
                    running_loss_2 += loss2.item()
                    running_loss_3 += loss3.item()
                    running_loss_4 += loss4.item()
                    if i % log_interval == log_interval - 1:  # print every 2000 mini-batches
                        if log != None:
                            log["total"].append(running_loss_total / log_interval)
                            log["task_0"].append(running_loss_1 / log_interval)
                            log["task_1"].append(running_loss_2 / log_interval)
                            log["task_2"].append(running_loss_3 / log_interval)
                            log["task_3"].append(running_loss_4 / log_interval)
                        print('[%d, %5d] loss: %.3f' %
                              (epoch + 1, i + 1, running_loss_total / log_interval))
                        running_loss_total = 0.0
                        running_loss_1 = 0.0
                        running_loss_2 = 0.0
                        running_loss_3 = 0.0
                        running_loss_4 = 0.0

        print('Finished Training')

    def test_overall(self, net, performance=None, cud=None):
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
                images, labels = data

                if cud != None:
                    images = images.cuda(cud)

                labels0 = torch.tensor([label_to_task0(l) for l in labels])
                labels1 = torch.tensor([label_to_task1(l) for l in labels])
                labels2 = torch.tensor([label_to_task2(l) for l in labels])
                labels3 = torch.tensor([label_to_task3(l) for l in labels])

                if cud != None:
                    labels0 = labels0.cuda(cud)
                    labels1 = labels1.cuda(cud)
                    labels2 = labels2.cuda(cud)
                    labels3 = labels3.cuda(cud)

                outputs0, outputs1, outputs2, outputs3 = net(images)
                _, predicted0 = torch.max(outputs0.data, 1)
                _, predicted1 = torch.max(outputs1.data, 1)
                _, predicted2 = torch.max(outputs2.data, 1)
                _, predicted3 = torch.max(outputs3.data, 1)
                total0 += labels0.size(0)
                total1 += labels1.size(0)
                total2 += labels2.size(0)
                total3 += labels3.size(0)

                correct0 += (predicted0 == labels0).sum().item()
                correct1 += (predicted1 == labels1).sum().item()
                correct2 += (predicted2 == labels2).sum().item()
                correct3 += (predicted3 == labels3).sum().item()

        print('task_0 accuracy of the network on the 10000 test images: %d %%' % (
                100 * correct0 / total0))
        print('task_1 accuracy of the network on the 10000 test images: %d %%' % (
                100 * correct1 / total1))
        print('task_2 accuracy of the network on the 10000 test images: %d %%' % (
                100 * correct2 / total2))
        print('task_3 accuracy of the network on the 10000 test images: %d %%' % (100 * correct3 / total3))

        for name, param in net.named_parameters():
            if param.requires_grad:
                if name == "g_logits":
                    print(name, param.data)
                    print(nn.functional.softmax(param.data[0], dim=0))
                    print(nn.functional.softmax(param.data[1], dim=0))
                    print(nn.functional.softmax(param.data[2], dim=0))

        if performance != None:
            performance["task_0"].append(100 * correct0 / total0)
            performance["task_1"].append(100 * correct1 / total1)
            performance["task_2"].append(100 * correct2 / total2)
            performance["task_3"].append(100 * correct3 / total3)
    # If no improvement after 3 iterations, probably bad initialization. Retry.
    # Every epoch, save to file.
    def get_performance_data(self, d, epochs=1, cud=None, i=0, retry=False):
        if cud != None:
            c_crit = nn.CrossEntropyLoss().cuda(cud)
        else:
            c_crit = nn.CrossEntropyLoss()
        # If files for performance, log, meta, and model already exist, use them.
        data_name = "gating/" + str(d) + "_iter" + str(i) + '.csv'
        log_name = "gating/log" + str(d) + "_iter" + str(i) + '.csv'
        meta_name = "gating/meta" + str(d) + "_iter" + str(i)

        if os.path.isfile(data_name) and os.path.isfile(log_name) and os.path.isfile(meta_name):
            print("loading model")
            checkpoint = torch.load(meta_name)
            retries = checkpoint["retries"]
            if retry == True:
                if retries >= 4:
                    return True
                retries += 1
                executed_epochs = 0
                net = Net()
                optimizer = optim.Adam(net.parameters())
                if cud != None:
                    net = net.cuda(cud)
                performance = {'task_0': [], 'task_1': [], 'task_2': [], 'task_3': []}
                df = pd.read_csv(log_name, dtype='float')  # you wanted float datatype
                log = df.to_dict(orient='list')
                del log['Unnamed: 0']
                log['total'].append(-1)
                log['task_0'].append(-1)
                log['task_1'].append(-1)
                log['task_2'].append(-1)
                log['task_3'].append(-1)

            else:
                executed_epochs = checkpoint["epochs"]
                net = Net()
                if cud != None:
                    net = net.cuda(cud)
                net.load_state_dict(checkpoint["model_state_dict"])
                optimizer = optim.Adam(net.parameters())
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

                df = pd.read_csv(data_name, dtype='float')  # you wanted float datatype
                performance = df.to_dict(orient='list')
                df = pd.read_csv(log_name, dtype='float')  # you wanted float datatype
                log = df.to_dict(orient='list')
                del log['Unnamed: 0']
                del performance['Unnamed: 0']

            print("retry #", retries)
            print("epoch #", executed_epochs)
            print("performance dict:", performance)
            print("log dict:", log)
        else:
            print("creating new model")
            net = Net()
            optimizer = optim.Adam(net.parameters())
            if cud != None:
                net.cuda(cud)
            performance = {'task_0': [], 'task_1': [], 'task_2': [], 'task_3': []}
            log = {'total': [], 'task_0': [], 'task_1': [], 'task_2': [], 'task_3': []}
            retries = 0
            executed_epochs = 0

        for e in range(executed_epochs, epochs):
            self.train(net, optimizer, c_crit, log=log, epochs=1, cud=cud)
            self.test_overall(net, performance, cud=cud)

            if e >= 5:
                if (performance['task_0'][-1] < 34 or performance['task_1'][-1] < 5
                        or performance['task_2'][-1] < 5 or performance['task_3'][-1] < 11):
                    return False

            print(performance)
            df = pd.DataFrame(data=performance)
            df.to_csv(data_name)
            df = pd.DataFrame(data=log)
            df.to_csv(log_name)

            torch.save({
                "retries": retries,
                "epochs": e + 1,
                "model_state_dict": net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()
            }, meta_name)

            print("Trained:", d, i)
        return True

    def train_job(self, data_counts, p_num, cud=None):
        for i in range(0, 5):
            for d in data_counts:
                trainset = Position('../cifar_data', train=True,
                                    download=True, data_portion=d)
                self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                                          shuffle=True, num_workers=2)
                testset = Position('../cifar_data', train=False,
                                   download=True, data_portion=d)
                self.testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                                         shuffle=False, num_workers=2)

                print("Train Data Count:", len(trainset))
                print("Test Data Count:", len(testset))

                print("get_performance_data ", d, i)
                result = self.get_performance_data(d, epochs=100, cud=cud, i=i)
                while result != True:
                    print("retrying")
                    result = self.get_performance_data(d, epochs=100, cud=cud, retry=True, i=i)

"""
Automated Testing
"""

if __name__ == '__main__':

    cuda0 = torch.device('cuda:0')
    #cuda1 = torch.device('cuda:1')
    #cuda2 = torch.device('cuda:2')
    #cuda3 = torch.device('cuda:3')

    #procs = []
    #print("Starting process 0")
    e = Executor()
    e.train_job([1/8, 1/4, 1/2, 1], 0, cuda0)
    #proc = Process(target=train_job, args=([[4, 4, 0], [4, 4, 1], [4, 4, 2], [4, 4, 3], [4, 4, 4]], 0, cuda0))
    #procs.append(proc)
    #proc.start()

    #for proc in procs:
    #    proc.join()
