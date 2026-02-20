from torchvision import transforms
from torchvision.datasets import CIFAR10
import torch
from src.data.base_datamodule import BaseDataModule, BaseDupDataModule
import numpy as np
import jax
from tqdm import tqdm
from torch.utils.data import random_split


class CIFAR10DataModule(BaseDataModule):
    def __init__(self, config, generator):
        augmentations = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
        normalize = [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            transforms.Lambda(
                lambda x: x.permute(1, 2, 0)
            ),  # Pytorch uses channels first, but we want channels last
        ]
        train_transform = transforms.Compose(augmentations + normalize)
        test_transform = transforms.Compose(normalize)
        super().__init__(config, generator, CIFAR10, train_transform, test_transform)


class CIFAR10FeatureDataModule(BaseDataModule):

    def __init__(self, config, generator):
        
        train_data = torch.load('cifar10_features/cifar10_train_features.pt')
        test_data = torch.load('cifar10_features/cifar10_test_features.pt')
        train_features = train_data['features']
        train_labels = train_data['labels']
        test_features = test_data['features']
        test_labels = test_data['labels']
        self.full_train_dataset = torch.utils.data.TensorDataset(train_features, train_labels)
        self.test_dataset = torch.utils.data.TensorDataset(test_features, test_labels)
        super().__init__(config, generator, None, None, None)

    def setup(self):
        val_size = int(len(self.full_train_dataset) * self.val_split)
        train_size = len(self.full_train_dataset) - val_size
        self.train_dataset, self.val_dataset = random_split(
            self.full_train_dataset,
            [train_size, val_size],
            generator=self.split_generator,
        )

        forget_size = int(train_size * self.forget_split)
        retain_size = train_size - forget_size
        self.retain_dataset, self.forget_dataset = random_split(
            self.train_dataset,
            [retain_size, forget_size],
            generator=self.split_generator,
        )

        print(f"{len(self.train_dataset)=}")
        print(f"{len(self.val_dataset)=}")
        print(f"{len(self.test_dataset)=}")
        print(f"{len(self.forget_dataset)=}")
        print(f"{len(self.retain_dataset)=}")


class CIFAR10DupDataModule(BaseDupDataModule):
    def __init__(self, config, generator, dup):
        print("Loading CIFAR10 dup dataset...")
        augmentations = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
        normalize = [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            transforms.Lambda(
                lambda x: x.permute(1, 2, 0)
            ),
        ]
        train_transform = transforms.Compose(augmentations + normalize)
        test_transform = transforms.Compose(normalize)

        self.full_train_dataset = CIFAR10(
            root=config["dataset"]["root"],
            train=True,
            download=True,
            transform=train_transform,
        )
        print(torch.Tensor(self.full_train_dataset.data).shape)
        print(torch.Tensor(self.full_train_dataset.targets).shape)

        self.test_dataset = CIFAR10(
            root=config["dataset"]["root"],
            train=False,
            download=True,
            transform=test_transform,
        )
        super().__init__(config, generator, None, None, None, dup)

    def setup(self):
        val_size = int(len(self.full_train_dataset) * self.val_split)
        train_size = len(self.full_train_dataset) - val_size
        experiment_dataset, self.val_dataset = random_split(
            self.full_train_dataset,
            [train_size, val_size],
            generator=self.split_generator,
        )
        
        forget_size = int(train_size * self.forget_split / (1 + self.forget_split))
        retain_size = train_size - forget_size * 2
        partial_retain_dataset, dup_dataset, extra_dataset = random_split(
            experiment_dataset,
            [retain_size, forget_size, forget_size],
            generator=self.split_generator,
        )
        self.retain_dataset = torch.utils.data.ConcatDataset([partial_retain_dataset, dup_dataset])

        if self.dup is True:
            self.train_dataset = torch.utils.data.ConcatDataset([self.retain_dataset, dup_dataset])
            self.forget_dataset = dup_dataset
        else:
            self.train_dataset = torch.utils.data.ConcatDataset([self.retain_dataset, extra_dataset])
            self.forget_dataset = extra_dataset

        print(f"{len(self.val_dataset)=}")
        print(f"{len(self.test_dataset)=}")
        print(f"{len(self.train_dataset)=}")
        print(f"{len(self.forget_dataset)=}")
        print(f"{len(self.retain_dataset)=}")


class CIFAR100FeatureDataModule(BaseDataModule):

    def __init__(self, config, generator):
        
        train_data = torch.load('cifar100_features/cifar100_train_features.pt')
        test_data = torch.load('cifar100_features/cifar100_test_features.pt')
        train_features = train_data['features']
        train_labels = train_data['labels']
        test_features = test_data['features']
        test_labels = test_data['labels']
        self.full_train_dataset = torch.utils.data.TensorDataset(train_features, train_labels)
        self.test_dataset = torch.utils.data.TensorDataset(test_features, test_labels)
        super().__init__(config, generator, None, None, None)

    def setup(self):
        val_size = int(len(self.full_train_dataset) * self.val_split)
        train_size = len(self.full_train_dataset) - val_size
        self.train_dataset, self.val_dataset = random_split(
            self.full_train_dataset,
            [train_size, val_size],
            generator=self.split_generator,
        )

        forget_size = int(train_size * self.forget_split)
        retain_size = train_size - forget_size
        self.retain_dataset, self.forget_dataset = random_split(
            self.train_dataset,
            [retain_size, forget_size],
            generator=self.split_generator,
        )

        print(f"{len(self.train_dataset)=}")
        print(f"{len(self.val_dataset)=}")
        print(f"{len(self.test_dataset)=}")
        print(f"{len(self.forget_dataset)=}")
        print(f"{len(self.retain_dataset)=}")
