from PIL import Image
import os
import os.path
import numpy as np
import sys
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
import matplotlib.pyplot as plt

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

class Superimposed(Dataset):
    img_size = 32
    base_folder_cifar = 'cifar-10-batches-py'
    train_list_cifar = [
        'data_batch_1',
        'data_batch_2',
        'data_batch_3',
        'data_batch_4',
        'data_batch_5',
    ]

    test_list_cifar = [
        'test_batch',
    ]
    meta_cifar = {
        'filename': 'batches.meta',
        'key': 'label_names'
    }
    training_file_mnist = 'training.pt'
    test_file_mnist = 'test.pt'
    mnist_classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']

    cifar_mnist_to_num = [[i for i in range(10*j, 10*(j+1))] for j in range(0, 10)]

    cifar_size = 500

    def __init__(self, mnist_root, cifar_root, train=True,
                 cifar_transform=None, cifar_target_transform=None,
                 mnist_transform=None, mnist_target_transform=None,
                 download=False):

        self.mnist_root = os.path.expanduser(mnist_root)
        self.cifar_root = os.path.expanduser(cifar_root)
        self.train = train
        self.init_cifar(cifar_transform, cifar_target_transform)
        self.init_mnist(mnist_transform, mnist_target_transform)
        self.to_tensor = transforms.ToTensor()
        return

    def init_cifar(self, cifar_transform, cifar_target_transform):
        self.cifar_transform = cifar_transform
        self.cifar_target_transform = cifar_target_transform

        if self.train:
            downloaded_list = self.train_list_cifar
        else:
            downloaded_list = self.test_list_cifar

        self.cifar_data = []
        self.cifar_targets = []

        # now load the picked numpy arrays
        for file_name in downloaded_list:
            file_path = os.path.join(self.cifar_root, self.base_folder_cifar, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')

                self.cifar_data.append(entry['data'])

                if 'labels' in entry:
                    self.cifar_targets.extend(entry['labels'])
                else:
                    self.cifar_targets.extend(entry['fine_labels'])

        self.cifar_data = np.vstack(self.cifar_data).reshape(-1, 3, 32, 32)
        self.cifar_data = self.cifar_data.transpose((0, 2, 3, 1))  # convert to HWC

        path = os.path.join(self.cifar_root, self.base_folder_cifar, self.meta_cifar['filename'])
        with open(path, 'rb') as infile:
            if sys.version_info[0] == 2:
                data = pickle.load(infile)
            else:
                data = pickle.load(infile, encoding='latin1')
            self.cifar_classes = data[self.meta_cifar['key']]
        self.cifar_class_to_idx = {_class: i for i, _class in enumerate(self.cifar_classes)}

    def init_mnist(self, mnist_transform, mnist_target_transform):
        self.mnist_raw_folder = os.path.join(self.mnist_root, 'raw')
        self.mnist_processed_folder = os.path.join(self.mnist_root, 'processed')
        self.mnist_class_to_idx = {_class: i for i, _class in enumerate(self.mnist_classes)}

        self.mnist_transform = mnist_transform
        self.mnist_target_transform = mnist_target_transform
        if self.train:
            data_file = self.training_file_mnist
        else:
            data_file = self.test_file_mnist

        self.mnist_data, self.mnist_targets = torch.load(os.path.join(self.mnist_processed_folder, data_file))

    def __getitem__(self, index):

        cifar_index = index
        mnist_index = index

        cifar_img, cifar_target = self.cifar_data[cifar_index], self.cifar_targets[cifar_index]

        cifar_img = Image.fromarray(cifar_img)

        if self.cifar_transform is not None:
            cifar_img = self.cifar_transform(img)

        if self.cifar_target_transform is not None:
            cifar_target = self.cifar_target_transform(target)

        mnist_img, mnist_target = self.mnist_data[mnist_index], int(self.mnist_targets[mnist_index])
        mnist_img = Image.fromarray(mnist_img.numpy(), mode='L')

        if self.mnist_transform is not None:
            mnist_img = self.mnist_transform(mnist_img)

        if self.mnist_target_transform is not None:
            mnist_target = self.mnist_target_transform(mnist_target)

        #Now we superimpose these images
        mnist_img = mnist_img.resize((self.img_size,self.img_size), Image.ANTIALIAS)
        mnist_img = mnist_img.convert("RGB")
        img = Image.blend(mnist_img, cifar_img, 0.5)
        target = torch.tensor([cifar_target, mnist_target])
        return self.to_tensor(img), target

    def __len__(self):
        return int(min(len(self.cifar_data), len(self.mnist_data))/8)
