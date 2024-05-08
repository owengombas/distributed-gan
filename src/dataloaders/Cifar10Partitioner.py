from typing import List, Tuple

import torch
import torch.utils.data
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

from dataloaders.DataPartitioner import DataPartitioner, _get_partition

cifar10_shape = (3, 32, 32)

class Cifar10Partitioner(DataPartitioner):
    """
    Partition CIFAR10 dataset
    """

    def __init__(self, world_size: int, rank: int, path: str = "data/cifar10"):
        self.world_size = world_size
        self.rank = rank
        self.cifar10_train = None
        self.cifar10_test = None
        self.path = path

    def load_data(self):
        transform = transforms.Compose(
            [
                # transforms.Resize(64),
                # transforms.CenterCrop(64),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        self.cifar10_train = CIFAR10(root=self.path, download=True, transform=transform)
        self.cifar10_test = CIFAR10(root=self.path, train=False, download=True, transform=transform)

    def get_train_partition(
        self, partition_id: int
    ) -> Tuple[torch.utils.data.Subset, int, int]:
        return _get_partition(self.world_size, partition_id, self.cifar10_train)

    def shuffle(self):
        self.cifar10_train = torch.utils.data.Subset(
            self.cifar10_train, torch.randperm(len(self.cifar10_train))
        )
        self.cifar10_test = torch.utils.data.Subset(
            self.cifar10_test, torch.randperm(len(self.cifar10_test))
        )

    def get_test_partition(
        self, partition_id: int
    ) -> Tuple[torch.utils.data.Subset, int, int]:
        return _get_partition(self.world_size, partition_id, self.cifar10_test)

    @property
    def train_dataset(self) -> torch.utils.data.Dataset:
        return self.cifar10_train

    @property
    def test_dataset(self) -> torch.utils.data.Dataset:
        return self.cifar10_test
