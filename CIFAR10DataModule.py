import os
import random
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
import torch

from components.constant import DATA_FOLDER_

class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, data_dir = DATA_FOLDER_ + "CIFAR10/", batch_size=64, train_val_split=0.8, random_seed=42, num_workers=None):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_val_split = train_val_split
        self.random_seed = random_seed
        self.num_workers = num_workers if num_workers is not None else os.cpu_count()
        self.num_classes = 10
        self.num_features = 32 * 32 * 3
        self.input_size = [32, 32, 3]
        self.train_dataset = None

        # Transformations to apply to the data
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

    def prepare_data(self):
        # Download the CIFAR-10 dataset if it doesn't exist
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage=None):

        if self.train_dataset is not None:
            return

        full_dataset = CIFAR10(self.data_dir, train=True, transform=self.transform, download=True)
        dataset_size = len(full_dataset)
        train_size = int(dataset_size * self.train_val_split)
        val_size = dataset_size - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(self.random_seed))
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.test_dataset = CIFAR10(self.data_dir, train=False, transform=self.transform, download=True)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers)