import os
import pickle
import random
import tarfile
import urllib.request
from typing import Callable, Optional, Union, List

from PIL import Image
from torchvision.datasets import CIFAR100, ImageFolder
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Subset
from datasets import load_dataset
from collections import defaultdict


class HuggingFaceWrapper(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.data = [(item['image'], item['label']) for item in hf_dataset if item['label'] < 190]
        self.transform = transform
        self.targets = [label for _, label in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


class CUBDataset(Dataset):
    def __init__(self, root_dir, train=True, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        images_file = os.path.join(root_dir, 'images.txt')
        labels_file = os.path.join(root_dir, 'image_class_labels.txt')
        split_file = os.path.join(root_dir, 'train_test_split.txt')

        with open(images_file) as f:
            id_to_path = {int(line.split()[0]): line.split()[1] for line in f}
        with open(labels_file) as f:
            id_to_label = {int(line.split()[0]): int(line.split()[1]) - 1 for line in f}
        with open(split_file) as f:
            id_to_split = {int(line.split()[0]): int(line.split()[1]) for line in f}

        self.samples = [
            (os.path.join(root_dir, 'images', id_to_path[i]), id_to_label[i])
            for i in id_to_path
            if (id_to_split[i] == 1 if train else id_to_split[i] == 0)
        ]
        self.targets = [label for _, label in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


class ImageFolderWithTargets(ImageFolder):
    def find_classes(self, directory):
        classes = [d.name for d in os.scandir(directory) if d.is_dir() and not d.name.startswith('.')]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def __init__(self, root, transform=None, split="train"):
        super().__init__(root, transform=transform)

        class_to_indices = defaultdict(list)
        for i, (_, label) in enumerate(self.samples):
            class_to_indices[label].append(i)

        rng = random.Random(1234)
        
        selected_indices = []
        for label, indices in class_to_indices.items():
            rng.shuffle(indices)
            split_point = int(0.8 * len(indices))
            if split == "train":
                selected_indices.extend(indices[:split_point])
            else:
                selected_indices.extend(indices[split_point:])

        self.samples = [self.samples[i] for i in selected_indices]
        self.targets = [sample[1] for sample in self.samples]


class TinyImageNetDataset(Dataset):
    def __init__(self, root_dir, train=True, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        wnids_path = os.path.join(root_dir, 'wnids.txt')
        with open(wnids_path, 'r') as f:
            wnids = [line.strip() for line in f]
        class_to_idx = {wnid: idx for idx, wnid in enumerate(wnids)}

        if train:
            train_dir = os.path.join(root_dir, 'train')
            for wnid in os.listdir(train_dir):
                if wnid not in class_to_idx:
                    continue
                class_idx = class_to_idx[wnid]
                images_folder = os.path.join(train_dir, wnid, 'images')
                for img_file in os.listdir(images_folder):
                    img_path = os.path.join(images_folder, img_file)
                    self.samples.append((img_path, class_idx))
        else:
            val_dir = os.path.join(root_dir, 'val')
            annotations_file = os.path.join(val_dir, 'val_annotations.txt')
            images_folder = os.path.join(val_dir, 'images')

            img_to_wnid = {}
            with open(annotations_file, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    img_file, wnid = parts[0], parts[1]
                    img_to_wnid[img_file] = class_to_idx[wnid]

            for img_file in os.listdir(images_folder):
                if img_file in img_to_wnid:
                    img_path = os.path.join(images_folder, img_file)
                    self.samples.append((img_path, img_to_wnid[img_file]))

        self.targets = [label for _, label in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


class DatasetFactory:
    @staticmethod
    def download_cub(data_dir):
        url = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1"
        target_path = os.path.join(data_dir, "CUB_200_2011.tgz")
        extract_path = os.path.join(data_dir, "CUB_200_2011")
        if not os.path.exists(extract_path):
            print("Downloading CUB dataset...")
            urllib.request.urlretrieve(url, target_path)
            with tarfile.open(target_path, "r:gz") as tar:
                tar.extractall(path=data_dir)
            os.remove(target_path)

    @staticmethod
    def download_imagenet_r(data_dir):
        url = "https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar"
        target_path = os.path.join(data_dir, "imagenet-r.tar")
        extract_path = os.path.join(data_dir, "imagenet-r")
        if not os.path.exists(extract_path):
            print("Downloading ImageNet-R dataset...")
            urllib.request.urlretrieve(url, target_path)
            with tarfile.open(target_path, "r:") as tar:
                tar.extractall(path=data_dir)
            os.remove(target_path)

    @staticmethod
    def download_imagenet_a(data_dir):
        url = "https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar"
        target_path = os.path.join(data_dir, "imagenet-a.tar")
        extract_path = os.path.join(data_dir, "imagenet-a")
        if not os.path.exists(extract_path):
            print("Downloading ImageNet-A dataset...")
            urllib.request.urlretrieve(url, target_path)
            with tarfile.open(target_path, "r:") as tar:
                tar.extractall(path=data_dir)
            os.remove(target_path)

    @staticmethod
    def get_dataset(name: str, split: str, transform: Optional[Callable] = None, data_dir: str = "./data") -> Dataset:
        name = name.lower()
        split = split.lower()
        train = split == 'train'

        if name == "cifar100":
            dataset = CIFAR100(root=data_dir, train=train, download=True, transform=transform)
            return dataset

        elif name == "cars":
            hf_dataset = load_dataset("tanganke/stanford_cars", split=split)
            return HuggingFaceWrapper(hf_dataset, transform=transform)

        elif name == "imagenet-r":
            DatasetFactory.download_imagenet_r(data_dir)
            return ImageFolderWithTargets(root=os.path.join(data_dir, 'imagenet-r'), transform=transform, split=split)

        elif name == "imagenet-a":
            DatasetFactory.download_imagenet_a(data_dir)
            return ImageFolderWithTargets(root=os.path.join(data_dir, 'imagenet-a'), transform=transform, split=split)

        elif name == "cub":
            DatasetFactory.download_cub(data_dir)
            return CUBDataset(root_dir=os.path.join(data_dir, 'CUB_200_2011'), train=train, transform=transform)

        elif name == "t-imagenet":
            return TinyImageNetDataset(root_dir=os.path.join(data_dir, 'tiny-imagenet-200'), train=train, transform=transform)

        else:
            raise ValueError(f"Unsupported dataset: {name}")


class UnifiedDataLoader:
    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        data_dir: str = "./data",
        is_class_incremental: bool = False,
        n_tasks: int = 10,
        seed: int = 42,
        batch_size: int = 64,
        shuffle: bool = True,
        num_workers: int = 4,
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.transform = transform
        self.data_dir = data_dir
        self.is_class_incremental = is_class_incremental
        self.n_tasks = n_tasks
        self.seed = seed
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

        self.dataset = DatasetFactory.get_dataset(dataset_name, split, transform, data_dir)

    def get_dataloaders(self) -> Union[DataLoader, List[DataLoader]]:
        if not self.is_class_incremental:
            return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)

        # standard class-incremental task split
        class_to_indices = defaultdict(list)
        for idx, label in enumerate(self.dataset.targets):
            class_to_indices[label].append(idx)

        classes = sorted(class_to_indices.keys())
        random.seed(self.seed)
        random.shuffle(classes)

        n_total = len(classes)
        if n_total % self.n_tasks != 0:
            raise ValueError(f"Number of classes ({n_total}) is not divisible by number of tasks ({self.n_tasks}).")

        task_class_lists = [classes[i::self.n_tasks] for i in range(self.n_tasks)]

        loaders = []
        for class_subset in task_class_lists:
            subset_indices = [idx for cls in class_subset for idx in class_to_indices[cls]]
            subset = Subset(self.dataset, subset_indices)
            loader = DataLoader(subset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)
            loaders.append(loader)

        return loaders
