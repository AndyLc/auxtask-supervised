from PIL import Image
import os
import os.path
import numpy as np
import sys
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
import random

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

class Position(Dataset):
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

    cifar_size = 500

    def __init__(self, cifar_root, train=True,
                 cifar_transform=None, cifar_target_transform=None,
                 download=False, data_portion=1):

        self.cifar_root = os.path.expanduser(cifar_root)
        self.train = train
        self.data_portion = data_portion

        if not (os.path.isfile("data/train_mixed_data.pickle") and os.path.isfile("data/test_mixed_data.pickle")):
            self.init_cifar(cifar_transform, cifar_target_transform)
            self.init_square()
            self.generate_mixed_data()

        if self.train:
            with open("data/train_mixed_data.pickle", "rb") as handle:
                data = pickle.load(handle)
        else:
            with open("data/test_mixed_data.pickle", "rb") as handle:
                data = pickle.load(handle)

        self.data = data["data"]
        self.targets = data["targets"]
        self.to_tensor = transforms.ToTensor()

    def init_cifar(self, cifar_transform, cifar_target_transform):
        self.cifar_transform = cifar_transform
        self.cifar_target_transform = cifar_target_transform

        #if self.train:
        #    downloaded_list = self.train_list_cifar
        #else:
        #    downloaded_list = self.test_list_cifar
        downloaded_list = self.train_list_cifar + self.test_list_cifar

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

    def init_square(self):
        self.square_data = []
        self.square_targets = []
        for i in range(2, 30):
            for j in range(2, 30):
                for c in [0, 1, 2]:
                    img = []

                    for y in range(32):
                        row = []
                        for x in range(32):
                            if y >= j - 2 and y <= j + 2 and x >= i - 2 and x <= i + 2:
                                if c == 0:
                                    row.append(np.array([255, 0, 0]))
                                elif c == 1:
                                    row.append(np.array([0, 255, 0]))
                                else:
                                    row.append(np.array([0, 0, 255]))
                            else:
                                row.append(np.array([0, 0, 0]))

                        img.append(np.vstack(row))

                    self.square_data.append(np.array(img, dtype=np.uint8))
                    self.square_targets.append((c, [i-2, j-2]))

        c = list(zip(self.square_data, self.square_targets))
        random.shuffle(c)
        self.square_data, self.square_targets = zip(*c)

    def generate_mixed_data(self):
        all_data = []
        all_targets = []
        for index in range(len(self.square_data) * 35):
            color, loc = self.square_targets[index % len(self.square_data)]
            img = Image.fromarray(self.cifar_data[index % len(self.cifar_data)])
            square_img = Image.fromarray(self.square_data[index % len(self.square_data)])
            img = Image.blend(img, square_img, 0.5)
            label = [color, loc[0], loc[1], self.cifar_targets[index % len(self.cifar_data)]]
            all_data.append(np.asarray(img))
            all_targets.append(label)

        with open("data/train_mixed_data.pickle", "wb") as handle:
            pickle.dump({"data": all_data[:int(len(all_data) * 0.8)], "targets": all_targets[:int(len(all_data) * 0.8)]}, handle)

        with open("data/test_mixed_data.pickle", "wb") as handle:
            pickle.dump({"data": all_data[int(len(all_data) * 0.8):], "targets": all_targets[int(len(all_data) * 0.8):]}, handle)

    def __getitem__(self, index):
        img = Image.fromarray(self.data[index])
        target = torch.tensor(self.targets[index])
        return self.to_tensor(img), target

    def __len__(self):
        return int(len(self.data) * self.data_portion)
