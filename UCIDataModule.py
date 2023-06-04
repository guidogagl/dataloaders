from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from components.UCIDatasetLoader import UCIDatasetLoader
from components.TabularDataset import TabularDataset
from components.constant import *

import os
import pytorch_lightning as pl

class UCIDataModule(LightningDataModule):

    def __init__(
        self,
        dataset : str = "anneal",
        train_val_test_split: Tuple[float, float, float] = (0.75, 0.1, .15),
        batch_size: int = 64,
        num_workers: int = os.cpu_count(),
        scaler = StandardScaler(), # used to scale the data, must have a fit and a trasform method, None to not scale the data
    ):

        super().__init__()
        self.save_hyperparameters(logger=False)

        self.le = None
        self.scl = scaler

        training_features, _, training_labels, _, classes_number, _ = UCIDatasetLoader.uci_dataset_handler( dataset, train_val_test_split[2])
        
        self.num_classes = classes_number
        self.n_features = int(training_features.shape[1])

        self.num_classes = 10
        self.num_features = 32 * 32 * 3
        self.input_size = [32, 32, 3]
        self.train_dataset = None

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def get_num_classes(self):
        return self.num_classes

    def num_features(self):
        return self.n_features
    
    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        if not self.data_train and not self.data_val and not self.data_test:

            X_train, X_test, y_train, y_test, _, _ = UCIDatasetLoader.uci_dataset_handler( self.hparams.dataset, self.hparams.train_val_test_split[2])

            self.le = LabelEncoder().fit(y_train)
            y_train = self.le.transform(y_train)
            y_test = self.le.transform(y_test)

            X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=self.hparams.train_val_test_split[1])
            
            if self.scl is not None:
                self.scl.fit(X_train)
                X_train, X_valid, X_test = self.scl.transform(X_train), self.scl.transform(X_valid), self.scl.transform(X_valid)

            X_train, X_valid, X_test = torch.tensor(X_train).float(), torch.tensor(X_valid).float(), torch.tensor(X_valid).float()
            y_train, y_valid, y_test = torch.tensor(y_train).long(), torch.tensor(y_valid).long(), torch.tensor(y_valid).long()

            self.data_train, self.data_val, self.data_test = TabularDataset(X_train, y_train), TabularDataset(X_valid, y_valid), TabularDataset(X_test, y_test) 


    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, dataset= "anneal", 
        batch_size=64, 
        train_val_test_split: Tuple[float, float, float] = (0.75, 0.1, .15),
        num_workers = None,
        transform = StandardScaler()
        ):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.train_val_test_split = train_val_test_split
        
        self.num_workers = num_workers if num_workers is not None else os.cpu_count()
        self.num_classes = None
        self.num_features = None
        self.input_size =  None 
        self.train_dataset = None
        
        self.transform = transform

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if self.train_dataset is not None:
            return
                
        X_train, X_test, y_train, y_test, _, _ = UCIDatasetLoader.uci_dataset_handler( self.dataset, self.train_val_test_split[2])

        self.le = LabelEncoder().fit(y_train)
        y_train = self.le.transform(y_train)
        y_test = self.le.transform(y_test)

        self.num_classes = len( self.le.classes_ )

        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=self.train_val_test_split[1])
        
        if self.scl is not None:
            self.scl.fit(X_train)
            X_train, X_valid, X_test = self.scl.transform(X_train), self.scl.transform(X_valid), self.scl.transform(X_valid)

        X_train, X_valid, X_test = torch.tensor(X_train).float(), torch.tensor(X_valid).float(), torch.tensor(X_valid).float()
        y_train, y_valid, y_test = torch.tensor(y_train).long(), torch.tensor(y_valid).long(), torch.tensor(y_valid).long()

        self.num_features = int(X_train.size(1).item())
        self.input_size = self.num_features
        
        self.data_train, self.data_val, self.data_test = TabularDataset(X_train, y_train), TabularDataset(X_valid, y_valid), TabularDataset(X_test, y_test) 

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers)