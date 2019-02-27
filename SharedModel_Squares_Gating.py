import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('tkagg')
from noise_position import Position
import torch.nn as nn
import torch.nn.functional as F
import time
import torch.optim as optim

# # Loading Data
# 
# The output of torchvision datasets are PILImage images of range [0, 1].
# 
# We transform them to Tensors of normalized range [-1, 1].

# In[3]:


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = Position('../cifar_data', train=True,
                                        download=True, data_portion=0.125)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = Position('../cifar_data', train=False,
                                        download=True, data_portion=0.125)

testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

print("Train Data Count:", len(trainset))
print("Test Data Count:", len(testset))


# Here we ensure that there is no correlation between numbers and objects in our dataset.

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


# ## Training Data Samples

# In[10]:


# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    #plt.imshow(np.transpose(npimg, (1, 2, 0)))
    #plt.show()

# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))

# print labels
print(' '.join('%5s' % task0_classes[label_to_task0(labels[j])] for j in range(4)))
print(' '.join('%5s' % task1_classes[label_to_task1(labels[j])] for j in range(4)))
print(' '.join('%5s' % task2_classes[label_to_task2(labels[j])] for j in range(4)))
print(' '.join('%5s' % task3_classes[label_to_task3(labels[j])] for j in range(4)))

# # Defining the Model
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

# # Training the Network

# In[14]:


def train(net, optimizer, c_crit, log=None, cud=None, epochs=1):
    print("Starting Training")
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss_total = 0.0
        for i, data in enumerate(trainloader, 0):
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
                labels1.cuda(cud)
                labels2.cuda(cud)
                labels3.cuda(cud)
                labels4.cuda(cud)
                
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs1, outputs2, outputs3, outputs4 = net(inputs)
            loss1 = c_crit(outputs1, labels1)
            loss2 = c_crit(outputs2, labels2)
            loss3 = c_crit(outputs3, labels3)
            loss4 = c_crit(outputs4, labels4)

            """
            # Entropy regularization
            for name, param in net.named_parameters():
                if param.requires_grad:
                    if name == "g_logits":
                        #print(name, param.data)
                        distr0 = nn.functional.softmax(param.data[0], dim=0)
                        log0 = torch.mul(torch.log2(distr0), -1)
                        #print("distr0:", nn.functional.softmax(param.data[1]))
                        #print("logDistr:", torch.log2(distr0))
                        #print("negLogDistr:", log0)
                        #print("entropy:", distr0 * log0)
                        #print("sum:", torch.sum(distr0 * log0))
                        distr1 = nn.functional.softmax(param.data[0], dim=0)
                        log1 = torch.mul(torch.log2(distr1), -1)
                        #print(nn.functional.softmax(param.data[2], dim=0))
                        distr2 = nn.functional.softmax(param.data[0], dim=0)
                        log2 = torch.mul(torch.log2(distr2), -1)
                        entropy_loss = torch.sum(distr0 * log0) + torch.sum(distr1 * log1) + torch.sum(distr2 * log2)
            """

            loss = (loss1 + loss2 / 3 + loss3 / 3 + loss4 * 2) / 4# + 10 * entropy_loss
            loss.backward()
            optimizer.step()

            # print statistics
            with torch.no_grad():
                running_loss_total += loss.item()
                if i % 1000 == 999:    # print every 2000 mini-batches
                    if log != None:
                        log["total"].append(running_loss_total / 1000)
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss_total / 1000))
                    running_loss_total = 0.0

    print('Finished Training')

# Ensure that the network is as expected by visualization

# In[12]:

net = Net()
torch.onnx.export(net, dataiter.next()[0], "./gatingnet.onnx", verbose=True, export_params=True)

# ## Defining the Loss function and Optimizer

# In[13]:
classifier_criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())

train(net, optimizer, classifier_criterion)


# # Testing the Network

# In[171]:

dataiter = iter(testloader)
images, labels = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % task0_classes[label_to_task0(labels[j])] for j in range(4)))
print('GroundTruth: ', ' '.join('%5s' % task1_classes[label_to_task1(labels[j])] for j in range(4)))
print('GroundTruth: ', ' '.join('%5s' % task2_classes[label_to_task2(labels[j])] for j in range(4)))
print('GroundTruth: ', ' '.join('%5s' % task2_classes[label_to_task3(labels[j])] for j in range(4)))


# In[172]:

outputs1, outputs2, outputs3, outputs4 = net(images)

_, predicted = torch.max(outputs1, 1)
print('Predicted: ', ' '.join('%5s' % task0_classes[predicted[j]]
                              for j in range(4)))
_, predicted = torch.max(outputs2, 1)
print('Predicted: ', ' '.join('%5s' % task1_classes[predicted[j]]
                              for j in range(4)))
_, predicted = torch.max(outputs3, 1)
print('Predicted: ', ' '.join('%5s' % task2_classes[predicted[j]]
                              for j in range(4)))
_, predicted = torch.max(outputs4, 1)
print('Predicted: ', ' '.join('%5s' % task3_classes[predicted[j]]
                              for j in range(4)))


def test_overall(net, performance=None):
    correct0 = 0
    total0 = 0
    correct1 = 0
    total1 = 0
    correct2 = 0
    total2 = 0
    correct3 = 0
    total3 = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            labels0 = torch.tensor([label_to_task0(l) for l in labels])
            labels1 = torch.tensor([label_to_task1(l) for l in labels])
            labels2 = torch.tensor([label_to_task2(l) for l in labels])
            labels3 = torch.tensor([label_to_task3(l) for l in labels])
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
    print('task_3 accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct3 / total3))
    
    if performance != None:
        performance["task_0"].append(100 * correct0 / total0)
        performance["task_1"].append(100 * correct1 / total1)
        performance["task_2"].append(100 * correct2 / total2)
        performance["task_2"].append(100 * correct3 / total3)


# In[174]:
test_overall(net)

for name, param in net.named_parameters():
    if param.requires_grad:
        if name == "g_logits":
            print(name, param.data)
            print(nn.functional.softmax(param.data[0], dim=0))
            print(nn.functional.softmax(param.data[1], dim=0))
            print(nn.functional.softmax(param.data[2], dim=0))