import dataclasses

import torchvision
import torchvision.transforms as transforms
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader


class Cifar10(LightningDataModule):
    def __init__(self):
        super().__init__()
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: [str] = None):
        if stage == "fit":
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.Resize(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            transform_test = transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            self.train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train)
            self.val_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=32, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=32, shuffle=False)



if __name__ == "__main__":
    torchvision.datasets.CIFAR10(root='./data', train=True, download=True)

    module = Cifar10()
    dataloader = module.train_dataloader()
    for batch in dataloader:
        print(batch)
        pass


