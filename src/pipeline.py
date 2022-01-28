"""Contains data processing and pipeline functions for training."""
import os

import numpy as np
import torch
import torchvision as tv
from torchvision import transforms
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from PIL import Image
from torch.utils.data import DataLoader, Dataset

#from torchvision.transforms import ColorJitter, Resize
#from torchvision.transforms.functional import hflip, vflip


class MoleMapDataset(Dataset):

    def __init__(self, image_paths, transform):
        self.image_paths = image_paths

        self.transform = transform

    def __len__(self):
        return len(self.image_paths)


    def __getitem__(self, idx):
        # Load Image #
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = self.transform(image)

        return image, image


class MoleMapDataModule(pl.LightningDataModule):

    def __init__(self, image_paths, batch_size, image_size, num_workers, persistent_workers):
        self.image_paths = image_paths
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.transform = transforms.Compose(
        [
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Is this ImageNet?
        ])


    def setup(self, stage=None):
        train_paths, val_paths = train_test_split(self.image_paths, test_size=0.2)
        self.train_dataset = MoleMapDataset(train_paths, self.transform)
        self.val_dataset = MoleMapDataset(val_paths, self.transform)


    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers)

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers)
