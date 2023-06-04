import os
import random
import shutil
import requests
import tarfile
import numpy as np
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
import torch

class UCRDataModule(pl.LightningDataModule):
    def __init__(self, data_dir='./data', dataset_name='ECG200', batch_size=64, train_val_split=0.8, random_seed=42, num_workers=None):
        super().__init__()
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.train_val_split = train_val_split
        self.random_seed = random_seed
        self.num_workers = num_workers if num_workers is not None else os.cpu_count()
        self.num_classes = None
        self.num_features = None

    def prepare_data(self):
        # Download and extract dataset
        self.download()
        self.untar()


    def download(self):
        # Check if dataset directory exists, if not, create it
        dataset_dir = os.path.join(self.data_dir, self.dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)

        # Download the dataset file if it doesn't exist
        url = f'http://www.timeseriesclassification.com/Downloads/{self.dataset_name}.tar.gz'
        filename = os.path.join(dataset_dir, f'{self.dataset_name}.tar.gz')

        if not os.path.exists(filename):
            response = requests.get(url, stream=True)
            with open(filename, 'wb') as f:
                shutil.copyfileobj(response.raw, f)

    def untar(self):
        dataset_dir = os.path.join(self.data_dir, self.dataset_name)
        tar_filename = os.path.join(dataset_dir, f'{self.dataset_name}.tar.gz')

        # Extract the dataset if it hasn't been extracted yet
        if not os.path.exists(os.path.join(dataset_dir, self.dataset_name)):
            with tarfile.open(tar_filename, 'r:gz') as tar:
                tar.extractall(path=dataset_dir)

    def setup(self, stage=None):
        if self.train_data is not None:
            return  # setup already called

        # Load training dataset from file
        train_data_file = os.path.join(self.data_dir, self.dataset_name, self.dataset_name + '_TRAIN.npy')
        train_labels_file = os.path.join(self.data_dir, self.dataset_name, self.dataset_name + '_TRAIN.labels.npy')
        train_dataset = np.load(train_data_file)
        train_labels = np.load(train_labels_file)

        # Convert labels to integers
        unique_labels = np.unique(train_labels)
        label_to_index = {label: i for i, label in enumerate(unique_labels)}
        train_labels = np.array([label_to_index[label] for label in train_labels])

        # Split training dataset into training and validation sets
        dataset_size = len(train_dataset)
        train_size = int(dataset_size * self.train_val_split)
        val_size = dataset_size - train_size
        train_dataset, val_dataset = random_split(list(zip(train_dataset, train_labels)), [train_size, val_size], generator=torch.Generator().manual_seed(self.random_seed))
        
        # Separate features and labels
        self.train_data, self.train_labels = zip(*train_dataset)
        self.val_data, self.val_labels = zip(*val_dataset)

        # Load test dataset from file
        test_data_file = os.path.join(self.data_dir, self.dataset_name, self.dataset_name + '_TEST.npy')
        test_labels_file = os.path.join(self.data_dir, self.dataset_name, self.dataset_name + '_TEST.labels.npy')
        self.test_data = np.load(test_data_file)
        test_labels = np.load(test_labels_file)
        
        # Convert test labels to integers
        self.test_labels = np.array([label_to_index[label] for label in test_labels])

        # Set number of classes and features
        self.num_classes = len(unique_labels)
        self.num_features = self.train_data[0].shape[1]

    def train_dataloader(self):
        return DataLoader(list(zip(self.train_data, self.train_labels)), batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(list(zip(self.val_data, self.val_labels)), batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(list(zip(self.test_data, self.test_labels)), batch_size=self.batch_size, num_workers = self.num_workers)
