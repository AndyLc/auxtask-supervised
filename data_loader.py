from torch.utils import data
from torchvision import transforms as T
import random
from PIL import Image
import os
import os.path
import numpy as np
import sys
import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle


class CelebA(data.Dataset):
    """Dataset class for the CelebA dataset."""

    def __init__(self, image_dir, attr_path, selected_attrs, transform, mode, data_portion=1):
        """Initialize and preprocess the CelebA dataset."""
        self.data_portion = data_portion
        self.image_dir = image_dir
        self.attr_path = attr_path
        self.selected_attrs = selected_attrs
        self.transform = transform
        self.mode = mode
        self.train_dataset = []
        self.test_dataset = []
        self.attr2idx = {}
        self.idx2attr = {}
        self.preprocess()

        if mode == 'train':
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

    def preprocess(self):
        """Preprocess the CelebA attribute file."""
        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        all_attr_names = lines[1].split()
        print(all_attr_names)
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        lines = lines[2:]
        random.seed(1234)
        random.shuffle(lines)
        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            values = split[1:]

            label = []
            for attr_name in self.selected_attrs:
                idx = self.attr2idx[attr_name]
                label.append(values[idx] == '1')

            if (i+1) < 2000 * 20:
                self.test_dataset.append([filename, label])
            else:
                self.train_dataset.append([filename, label])

        print('Finished preprocessing the CelebA dataset...')

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
        filename, label = dataset[index]
        image = Image.open(os.path.join(self.image_dir, filename))
        return self.transform(image), torch.FloatTensor(label)

    def __len__(self):
        """Return the number of images."""
        return int(self.num_images * self.data_portion)

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

    def __init__(self, root, train=True,
                 cifar_transform=None, cifar_target_transform=None,
                 download=False, data_portion=1):

        self.root = os.path.expanduser(root)
        self.train = train
        self.data_portion = data_portion

        if not (os.path.isfile(root + "/train_mixed_data.pickle") and os.path.isfile(root + "/test_mixed_data.pickle")):
            self.init_cifar(cifar_transform, cifar_target_transform)
            self.init_square()
            self.generate_mixed_data()

        if self.train:
            with open(root + "/train_mixed_data.pickle", "rb") as handle:
                data = pickle.load(handle)
        else:
            with open(root + "/test_mixed_data.pickle", "rb") as handle:
                data = pickle.load(handle)

        self.data = data["data"]
        self.targets = data["targets"]
        self.to_tensor = transforms.ToTensor()

    def init_cifar(self, cifar_transform, cifar_target_transform):
        self.cifar_transform = cifar_transform
        self.cifar_target_transform = cifar_target_transform
        downloaded_list = self.train_list_cifar + self.test_list_cifar

        self.cifar_data = []
        self.cifar_targets = []

        # now load the picked numpy arrays
        for file_name in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder_cifar, file_name)
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

        path = os.path.join(self.root, self.base_folder_cifar, self.meta_cifar['filename'])
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
            label = [color, random.randint(0, 27), random.randint(0, 27), self.cifar_targets[index % len(self.cifar_data)]]
            all_data.append(np.asarray(img))
            all_targets.append(label)

        with open(self.root + "/train_mixed_data.pickle", "wb") as handle:
            pickle.dump({"data": all_data[:int(len(all_data) * 0.8)], "targets": all_targets[:int(len(all_data) * 0.8)]}, handle)

        with open(self.root + "/test_mixed_data.pickle", "wb") as handle:
            pickle.dump({"data": all_data[int(len(all_data) * 0.8):], "targets": all_targets[int(len(all_data) * 0.8):]}, handle)

    def __getitem__(self, index):
        img = Image.fromarray(self.data[index])
        target = torch.tensor(self.targets[index])
        return self.to_tensor(img), target

    def __len__(self):
        return int(len(self.data) * self.data_portion)


"""

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

"""


def get_loader(path, crop_size=178, image_size=128, batch_size=32, dataset='Superimposed', mode='train', num_workers=2, data_portion=1):
    transform = []
    if mode == 'train':
        transform.append(T.RandomHorizontalFlip())
    transform.append(T.CenterCrop(crop_size))
    transform.append(T.Resize(image_size))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)
    if dataset == 'CelebA':
        dataset = CelebA(path + "/images", path + "/list_attr_celeba.txt",
                         ['Bushy_Eyebrows', 'No_Beard', 'Brown_Hair', 'Blond_Hair', 'Black_Hair', 'Eyeglasses'], transform, mode, data_portion=data_portion)
    elif dataset == 'Superimposed':
        dataset = Position(path, train=(mode=='train'), data_portion=data_portion)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode=='train'),
                                  num_workers=num_workers)
    return data_loader

