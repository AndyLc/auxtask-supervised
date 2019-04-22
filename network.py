import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from task import Task
import unittest
from gated_layer import SamplingLayer, BlendingLayer, VILayer

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size()[0], -1)

class Network(nn.Module):
    def __init__(self, tasks, trainloader, testloader, config=None, log=None, cuda=None):
        super(Network, self).__init__()
        self.cuda = cuda
        self.tasks = tasks
        self.trainloader = trainloader
        self.testloader = testloader
        self.reg = config["reg"]
        self.in_channels = config["in_channels"]
        self.blocks = config["blocks"]
        self.fc_in = config["fc_in"]
        self.options = config["options"]
        self.log = log
        self.config = config
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        if self.cuda:
            self.criterion.cuda(self.cuda)

    def setup(self):
        self.optimizer = optim.Adam(self.parameters())

    def train_epoch(self, log_interval=1000):
        print("Starting Training")
        if self.log:
            print("Starting Training", file=open(self.log, "a"))
        running_loss_total = 0.0
        for i, data in enumerate(self.trainloader, 0):
            # get the inputs
            if len(data) == 2:
                inputs, labels = data
                tasks = None
            else:
                inputs, tasks, labels = data

            if self.cuda is not None:
                inputs = inputs.cuda(self.cuda)
                labels = labels.cuda(self.cuda)
                if tasks is not None:
                    tasks = tasks.cuda(self.cuda)

            tasks = tasks[:, None]
            task_labels = torch.stack([torch.tensor(self.tasks[tasks[l]].label_to_task(labels[l]), dtype=torch.long) for l in range(len(labels))])
            task_labels = task_labels[:, None]

            if self.cuda is not None:
                task_labels = task_labels.cuda(self.cuda)

            # zero the parameter gradients
            self.optimizer.zero_grad()

            # get non-gate parameter loss
            losses = self.get_losses(inputs, tasks, task_labels)

            scales = [t.scale for t in self.tasks]
            net_loss = losses
            #net_loss = sum([losses[i] * scales[i] for i in range(len(self.tasks))]) / len(losses)

            # get gate loss
            loss = net_loss
            loss.backward()
            self.optimizer.step()

            # print statistics
            with torch.no_grad():
                running_loss_total += loss.item()
                if i % log_interval == log_interval - 1:
                    if self.log:
                        print('[%5d] loss: %.3f' % (i + 1, running_loss_total / log_interval), file=open(self.log, "a"))
                    else:
                        print('[%5d] loss: %.3f' % (i + 1, running_loss_total / log_interval))
                    running_loss_total = 0.0
                    for name, param in self.named_parameters():
                        if param.requires_grad:
                            if name in ["g_logits", "q_logits"]:
                                if self.log:
                                    print(name, file=open(self.log, "a"))
                                    for i in range(len(param.data)):
                                        print(list(torch.sigmoid(param.data[i])),
                                              file=open(self.log, "a"))
                                print(name)
                                for i in range(len(param.data)):
                                    print(list(torch.sigmoid(param.data[i])))

                            if name in ["weights"]:
                                if self.log:
                                    print(name, file=open(self.log, "a"))
                                    for i in range(len(param.data)):
                                        print(list(param.data[i]),
                                              file=open(self.log, "a"))
                                print(name)
                                for i in range(len(param.data)):
                                    print(list(param.data[i]))

        print('Finished Training')
        if self.log:
            print("Finished Training", file=open(self.log, "a"))

    def test_overall(self, sample=False, individual=False, naive_sample=False, mixture=False, VI=False):
        self.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for data in self.testloader:

                if len(data) == 2:
                    inputs, labels = data
                    tasks = None
                else:
                    inputs, tasks, labels = data

                if self.cuda is not None:
                    inputs = inputs.cuda(self.cuda)
                    labels = labels.cuda(self.cuda)
                    if tasks is not None:
                        tasks = tasks.cuda(self.cuda)

                tasks = tasks[:, None]
                task_labels = torch.stack(
                    [torch.tensor(self.tasks[tasks[l]].label_to_task(l), dtype=torch.long) for l in labels])
                task_labels = task_labels[:, None]
                if self.cuda is not None:
                    task_labels = task_labels.cuda(self.cuda)
                outputs, _, _ = self.forward(inputs, tasks, None)
                predicted = torch.argmax(nn.functional.softmax(outputs, dim=1), dim=1)
                total += len(task_labels)

                task_labels = task_labels.reshape(-1)
                correct += float(sum(predicted == task_labels))


        #total_losses = total_losses / i

        if self.log:
            print('overall accuracy of the network on the 10000 test images: %.2f %%' % (
                    100 * float(correct) / float(total)), file=open(self.log, "a"))

        print('overall accuracy of the network on the 10000 test images: %.2f %%' % (
                    100 * float(correct) / float(total)))

        self.train()


class Individual(Network):
    def __init__(self, tasks, trainloader, testloader, config=None, log=None, cuda=None):
        super().__init__(tasks, trainloader, testloader, config=config, log=log, cuda=cuda)

        self.convs = nn.ModuleList([nn.Sequential(
            nn.Conv2d(self.in_channels, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.BatchNorm2d(128, affine=False),
            Flatten()
        ) for _ in range(self.blocks)])

        self.fcs = nn.ModuleList([nn.Sequential(
            nn.Linear(self.fc_in, 16),
            nn.Linear(16, self.tasks[i].size)
        ) for i in range(len(self.tasks))])

        self.setup()

    def forward(self, input, emb, labels):
        """
        :param input: Nximg_size
        :param emb: Nx1
        :param labels: Nx1
        :return: out: NxO, log_qs: NxB, log_ps: NxB
        """
        N = input.shape[0]

        out = torch.stack([self.convs[i](input) for i in range(len(self.convs))])
        out = torch.stack([self.fcs[i](out[i]) for i in range(len(self.convs))])

        O = out.shape[-1]

        out = torch.gather(out, 0, emb[None, :, :].expand(1, -1, self.options))
        out = out.reshape(N, O)

        return out, 0, 0

    def get_losses(self, inputs, emb, labels):
        results, _, _ = self(inputs, emb, labels)
        labels = labels.view(-1)
        losses = self.criterion(results, labels).mean(dim=0)
        return losses

class OneLayer(Network):
    def __init__(self, tasks, trainloader, testloader, config=None, log=None, cuda=None):
        super().__init__(tasks, trainloader, testloader, config=config, log=log, cuda=cuda)

        self.config["w"] = self.get_weight
        self.config["g"] = self.get_g_logits
        self.config["q"] = self.get_q_logits

        self.convs = nn.ModuleList([nn.Sequential(
            nn.Conv2d(self.in_channels, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.BatchNorm2d(128, affine=False),
            Flatten()
        ) for _ in range(self.blocks)])

        self.weights = nn.Parameter(torch.randn([len(self.tasks), self.blocks])) #T X B
        self.g_logits = nn.Parameter(torch.zeros([len(self.tasks), self.blocks]))
        self.q_logits = nn.Parameter(torch.zeros([len(self.tasks), self.options, self.blocks]))
        #self.gated_layer = VILayer(self.convs, self.config)
        self.fcs = nn.ModuleList([nn.Sequential(
            nn.Linear(self.fc_in, 16),
            nn.Linear(16, self.tasks[i].size)
        ) for i in range(len(self.tasks))])

        self.setup()

    def get_weight(self, emb):
        # emb: N x 1 for tasks
        # out: N x B
        N = emb.shape[0]
        B = self.weights.shape[1]
        selected = torch.gather(self.weights[None, :, :].expand(N, -1, -1), 1,
                                emb[:, :, None].expand(-1, -1, B))  # N x 1 x B
        return selected.reshape(N, -1)

    def get_g_logits(self, emb):
        # emb: N x 1 for tasks
        # out: N x B
        N = emb.shape[0]
        B = self.g_logits.shape[1]

        selected = torch.gather(self.g_logits[None, :, :].expand(N, -1, -1), 1,
                                emb[:, :, None].expand(-1, -1, B))  # N x 1 x B
        return selected.reshape(N, -1)

    def get_q_logits(self, emb, task_labels):
        # emb: N x 1 for tasks
        # task_labels: N x 1
        # out: N x B
        N = emb.shape[0]
        B = self.q_logits.shape[2]
        O = self.q_logits.shape[1]
        selected_tasks = torch.gather(self.q_logits[None, :, :, :].expand(N, -1, -1, -1), 1,
                                      emb[:, :, None, None].expand(-1, -1, O, B))  # N x 1 x O x B

        selected_y = torch.gather(selected_tasks, 2,
                                  task_labels[:, :, None, None].expand(-1, -1, -1, B))  # N x 1 x 1 x B

        return selected_y.reshape(N, -1)

    def forward(self, input, emb, labels):
        """
        :param input: Nximg_size
        :param emb: Nx1
        :param labels: Nx1
        :return: out: NxO, log_qs: NxB, log_ps: NxB
        """
        N = input.shape[0]

        gated_out, log_gs, log_qs = self.gated_layer(input, emb, labels)
        out = torch.stack([f(gated_out) for f in self.fcs])

        O = out.shape[-1]

        out = torch.gather(out, 0, emb[None, :, :].expand(1, -1, self.options))
        out = out.reshape(N, O)

        return out, log_gs, log_qs

    def get_losses(self, inputs, emb, labels):
        results, log_probs, extra_loss = self(inputs, emb, labels)
        labels = labels.reshape(-1)

        if log_probs is not 0:
            log_probs = log_probs.reshape(-1, 1)
        if extra_loss is not 0:
            extra_loss = extra_loss.reshape(-1, 1)

        loss = self.criterion(results, labels).reshape(-1, 1)
        return (loss + (loss.detach() * log_probs) + extra_loss).mean(dim=0)

class OneLayerBlend(OneLayer):
    def __init__(self, tasks, trainloader, testloader, config=None, log=None, cuda=None):
        super().__init__(tasks, trainloader, testloader, config=config, log=log, cuda=cuda)
        self.gated_layer = BlendingLayer(self.convs, self.config)

class OneLayerSample(OneLayer):
    def __init__(self, tasks, trainloader, testloader, config=None, log=None, cuda=None):
        super().__init__(tasks, trainloader, testloader, config=config, log=log, cuda=cuda)
        self.gated_layer = SamplingLayer(self.convs, self.config)

class OneLayerVI(OneLayer):
    def __init__(self, tasks, trainloader, testloader, config=None, log=None, cuda=None):
        super().__init__(tasks, trainloader, testloader, config=config, log=log, cuda=cuda)
        self.gated_layer = VILayer(self.convs, self.config)


class TestOneLayerNetwork(unittest.TestCase):
    def test_one_layer(self):
        config = {
            "reparam": True,
            "discrete": False,
            "in_channels": 3,
            "blocks": 3
        }

        num_tasks = 5

        tasks = [Task([None for _ in range(num_tasks)], None) for _ in range(num_tasks)]
        net = OneLayer(tasks, None, None, config=config)
        emb = torch.stack([torch.tensor([1]), torch.tensor([3]), torch.tensor([2])])

        net.g_logits[1][2] = 3
        net.g_logits[2][1] = 2
        net.g_logits[3][0] = 5
        answer = np.array([[0, 0, 3], [5, 0, 0], [0, 2, 0]])
        self.assertTrue((answer == net.get_g_logits(emb).detach().numpy()).all())

        net.q_logits[1][0][2] = 3
        net.q_logits[2][1][1] = 2
        net.q_logits[3][0][0] = 5
        answer = np.array([[0, 0, 3], [0, 0, 0], [0, 2, 0]])
        task_labels = torch.stack([torch.tensor([0]), torch.tensor([1]), torch.tensor([1])])
        self.assertTrue((answer == net.get_q_logits(emb, task_labels).detach().numpy()).all())

if __name__ == '__main__':
    unittest.main()