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
import torchvision.datasets as datasets
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

class CIFAR100(Dataset):
    def __init__(self, path, train=True, data_portion=1, transform=None):
        self.data_portion = data_portion
        self.train = train
        self.dataset = datasets.CIFAR100(root=path, train=train, download=True, transform=transform)
        self.fine_to_coarse = {0: 4, 1: 1, 2: 14, 3: 8, 4: 0, 5: 6, 6: 7, 7: 7, 8: 18, 9: 3, 10: 3, 11: 14, 12: 9, 13: 18, 14: 7, 15: 11, 16: 3, 17: 9, 18: 7, 19: 11, 20: 6, 21: 11, 22: 5, 23: 10, 24: 7, 25: 6, 26: 13, 27: 15, 28: 3, 29: 15, 30: 0, 31: 11, 32: 1, 33: 10, 34: 12, 35: 14, 36: 16, 37: 9, 38: 11, 39: 5, 40: 5, 41: 19, 42: 8, 43: 8, 44: 15, 45: 13, 46: 14, 47: 17, 48: 18, 49: 10, 50: 16, 51: 4, 52: 17, 53: 4, 54: 2, 55: 0, 56: 17, 57: 4, 58: 18, 59: 17, 60: 10, 61: 3, 62: 2, 63: 12, 64: 12, 65: 16, 66: 12, 67: 1, 68: 9, 69: 19, 70: 2, 71: 10, 72: 0, 73: 1, 74: 16, 75: 12, 76: 9, 77: 13, 78: 15, 79: 13, 80: 16, 81: 19, 82: 2, 83: 4, 84: 6, 85: 19, 86: 5, 87: 5, 88: 8, 89: 19, 90: 18, 91: 1, 92: 2, 93: 15, 94: 6, 95: 0, 96: 17, 97: 8, 98: 14, 99: 13}
        self.fine_classes = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle',
                         'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar',
                         'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile',
                         'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
                         'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster',
                         'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange',
                         'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy',
                         'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal',
                         'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar',
                         'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor',
                         'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
                         'worm']

        self.fine_to_label = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 1, 8: 0, 9: 0, 10: 1, 11: 1, 12: 0, 13: 1, 14: 2, 15: 0, 16: 2, 17: 1, 18: 3, 19: 1, 20: 1, 21: 2, 22: 0, 23: 0, 24: 4, 25: 2, 26: 0, 27: 0, 28: 3, 29: 1, 30: 1, 31: 3, 32: 1, 33: 1, 34: 0, 35: 2, 36: 0, 37: 2, 38: 4, 39: 1, 40: 2, 41: 0, 42: 1, 43: 2, 44: 2, 45: 1, 46: 3, 47: 0, 48: 2, 49: 2, 50: 1, 51: 1, 52: 1, 53: 2, 54: 0, 55: 2, 56: 2, 57: 3, 58: 3, 59: 3, 60: 3, 61: 4, 62: 1, 63: 1, 64: 2, 65: 2, 66: 3, 67: 2, 68: 3, 69: 1, 70: 2, 71: 4, 72: 3, 73: 3, 74: 3, 75: 4, 76: 4, 77: 2, 78: 3, 79: 3, 80: 4, 81: 2, 82: 3, 83: 4, 84: 3, 85: 3, 86: 3, 87: 4, 88: 3, 89: 4, 90: 4, 91: 4, 92: 4, 93: 4, 94: 4, 95: 4, 96: 4, 97: 4, 98: 4, 99: 4}
        self.super_classes = ['aquatic mammals', 'fish', 'flowers', 'food containers', 'fruit and vegetables',
                         'household electrical devices', 'household furniture', 'insects', 'large carnivores',
                         'large man-made outdoor things', 'large natural outdoor scenes',
                         'large omnivores and herbivores', 'medium-sized mammals', 'non-insect invertebrates', 'people',
                         'reptiles', 'small mammals', 'trees', 'vehicles 1', 'vehicles 2']

        if train == True:
            self.dataset = list(self.dataset)
            np.random.shuffle(self.dataset)

    def __getitem__(self, index):
        data, label = self.dataset.__getitem__(index)
        #inputs, tasks, labels
        return data, torch.tensor(self.fine_to_coarse[label]), torch.tensor(self.fine_to_label[label])

    def __len__(self):
        if self.train == True:
            return int(len(self.dataset) * self.data_portion)
        else:
            return len(self.dataset)


class MNIST(Dataset):
    def __init__(self, path, train=True, data_portion=1, transform=None):
        self.data_portion = data_portion
        self.train = train
        self.dataset = datasets.MNIST(root=path, train=train, download=True, transform=transform)

        if train == True:
            self.dataset = list(self.dataset)
            np.random.shuffle(self.dataset)

        self.super_classes = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
        self.class_one = ['zero', 'one', 'two', 'three', 'four', 'five', 'six']
        self.class_two = ['three', 'four', 'five', 'six', 'seven', 'eight', 'nine']

    def __getitem__(self, index):
        data, label = self.dataset.__getitem__(index)
        if label <= 2:
            task = 0
        elif label >= 7:
            task = 1
            label = label - 3
        else: # 3, 4, 5, 6, shared between the two.
            task = random.randint(0, 1)
            if task == 1:
                label = label - 3

        return data, torch.tensor(task), torch.tensor(label)

    def __len__(self):
        if self.train == True:
            return int(len(self.dataset) * self.data_portion)
        else:
            return len(self.dataset)


class RegularMNIST(Dataset):
    def __init__(self, path, train=True, data_portion=1, transform=None):
        self.train = train
        self.data_portion = data_portion
        self.dataset = datasets.MNIST(root=path, train=train, download=True, transform=transform)
        self.super_classes = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']

    def __getitem__(self, index):
        i = index // len(self.super_classes)
        data, label = self.dataset.__getitem__(i)
        l = torch.zeros([len(self.super_classes)])
        l[label] = 1
        return data, l

    def __len__(self):
        if self.train == True:
            return int(len(self.dataset) * self.data_portion)
        else:
            return len(self.dataset)

class MITStates(Dataset):

    def __init__(self, path, train=True, data_portion=1, transform=None):
        super(MITStates, self).__init__()
        self.path = path
        self.transform = transform
        self.train = train
        self.data_portion = data_portion

        self.imgs = []

        self.train_imgs = []
        self.test_imgs = []

        from os import listdir
        for f in listdir(path + '/images'):
            if ' ' not in f:
                continue
            adj, noun = f.split()

            for file_path in listdir(path + '/images/' + f):
                assert (file_path.endswith('jpg'))
                self.imgs += [{
                    'file_path': path + '/images/' + f + '/' + file_path,
                    'captions': [f],
                    'adj': adj,
                    'noun': noun
                }]

        self.caption_index_init_()
        self.noun_adj_to_label = {}
        self.noun_adj_to_count = {}

        nouns = list(self.noun2adjs.keys())
        self.noun_to_i = {nouns[i]: i for i in range(len(self.noun2adjs))}

        for noun, adjs in self.noun2adjs.items():
            names = list(adjs.keys())
            counts = list(adjs.values())
            self.noun_adj_to_label[noun] = {names[i]: i for i in range(len(adjs))}
            self.noun_adj_to_count[noun] = {names[i]: counts[i] for i in range(len(adjs))}

        seen = {}
        for img in self.imgs:
            if img['noun'] in self.noun2adjs and img['adj'] in self.noun2adjs[img['noun']]:
                if img['noun'] not in seen:
                    seen[img['noun']] = {}
                if img['adj'] not in seen[img['noun']]:
                    seen[img['noun']][img['adj']] = 1
                else:
                    seen[img['noun']][img['adj']] += 1

                if seen[img['noun']][img['adj']] <= self.noun_adj_to_count[img['noun']][img['adj']] * 0.8:
                    self.train_imgs.append(img)
                else:
                    self.test_imgs.append(img)

        if train is True:
            self.imgs = self.train_imgs
            np.random.shuffle(self.imgs)
        else:
            self.imgs = self.test_imgs

    def __getitem__(self, idx):
        noun = self.imgs[idx]['noun']
        adj = self.imgs[idx]['adj']
        return self.get_img(idx), self.noun_to_i[noun], self.noun_adj_to_label[noun][adj]

    def caption_index_init_(self):
        self.caption2imgids = {}
        self.noun2adjs = {}
        for i, img in enumerate(self.imgs):
            cap = img['captions'][0]
            adj = img['adj']
            noun = img['noun']
            if cap not in self.caption2imgids.keys():
                self.caption2imgids[cap] = []
            if noun not in self.noun2adjs.keys():
                self.noun2adjs[noun] = {}
            self.caption2imgids[cap].append(i)
            if adj not in self.noun2adjs[noun]:
                self.noun2adjs[noun][adj] = 1
            else:
                self.noun2adjs[noun][adj] += 1

        for noun, adjs in self.noun2adjs.items():
            assert len(adjs) >= 2

        filtered = {}
        for noun in self.noun2adjs:
            if len(self.noun2adjs[noun]) >= 5:
                filtered[noun] = {k: self.noun2adjs[noun][k] for k in list(self.noun2adjs[noun].keys())[:5]}

        self.noun2adjs = filtered

    def __len__(self):
        if self.train == True:
            return int(len(self.imgs) * self.data_portion)
        else:
            return len(self.imgs)

    def get_img(self, idx, raw_img=False):
        img_path = self.imgs[idx]['file_path']
        with open(img_path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        if raw_img:
            return img
        if self.transform:
            img = self.transform(img)
        return img

def get_loader(path, crop_size=178, image_size=32, batch_size=32, dataset='Superimposed', mode='train', num_workers=2, data_portion=1):
    transform = []

    #if mode == 'train' and dataset != 'MNIST':
    #    transform.append(T.RandomHorizontalFlip())
    #transform.append(T.CenterCrop(crop_size))
    transform.append(T.Resize(image_size))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)
    if dataset == 'CelebA':
        dataset = CelebA(path + "/images", path + "/list_attr_celeba.txt",
                         ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive'], transform, mode, data_portion=data_portion)
    elif dataset == 'Superimposed':
        dataset = Position(path, train=(mode=='train'), data_portion=data_portion)
    elif dataset == 'CIFAR100':
        dataset = CIFAR100(path, train=(mode=='train'), data_portion=data_portion, transform=transform)
    elif dataset == 'MNIST':
        dataset = MNIST(path, train=(mode=='train'), data_portion=data_portion, transform=transform)
    elif dataset == 'MITStates':
        dataset = MITStates(path, train=(mode=='train'), data_portion=data_portion, transform=transform)


    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode=='train'),
                                  num_workers=num_workers)
    return data_loader

