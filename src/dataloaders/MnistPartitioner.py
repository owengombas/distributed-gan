from typing import List, Tuple

import torch
import torch.utils.data
from torchvision.datasets import MNIST
import torchvision.transforms as transforms


from dataloaders.DataPartitioner import DataPartitioner, _get_partition

mnist_shape = (1, 28, 28)

class MnistPartitioner(DataPartitioner):
    """
    Partition MNIST dataset
    """

    def __init__(self, world_size: int, rank: int, path: str = "data/mnist"):
        self.world_size = world_size
        self.rank = rank
        self.mnist_train = None
        self.mnist_test = None
        self.path = path

    def load_data(self):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5), std=(0.5))
            ]
        )
        self.mnist_train = MNIST(root=self.path, download=True, transform=transform)
        self.mnist_test = MNIST(root=self.path, train=False, download=True, transform=transform)

    def shuffle(self):
        self.mnist_train = torch.utils.data.Subset(
            self.mnist_train, torch.randperm(len(self.mnist_train))
        )
        self.mnist_test = torch.utils.data.Subset(
            self.mnist_test, torch.randperm(len(self.mnist_test))
        )

    def get_train_partition(
        self, partition_id: int
    ) -> Tuple[torch.utils.data.Subset, int, int]:
        return _get_partition(self.world_size, partition_id, self.mnist_train)

    def get_test_partition(
        self, partition_id: int
    ) -> Tuple[torch.utils.data.Subset, int, int]:
        return _get_partition(self.world_size, partition_id, self.mnist_test)

    @property
    def train_dataset(self) -> torch.utils.data.Dataset:
        return self.mnist_train

    @property
    def test_dataset(self) -> torch.utils.data.Dataset:
        return self.mnist_test
