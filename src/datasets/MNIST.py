import torch
import torch.nn as nn
from typing import Tuple
from torchvision.datasets import MNIST
from torchvision import transforms
from datasets.DataPartitioner import DataPartitioner, _get_partition
from torchvision.datasets import MNIST
from torchvision import transforms
from typing import List, Tuple


SHAPE: Tuple[int, int, int] = (1, 28, 28)
NDF: int = 64
NGF: int = 64
Z_DIM: int = 100


class Partitioner(DataPartitioner):
    """
    Partition MNIST dataset
    """

    def __init__(self, world_size: int, rank: int, path: str = "data/mnist"):
        self.world_size = world_size
        self.rank = rank
        self.mnist_train = None
        self.mnist_test = None
        self.path = path

    def get_subset_from_indices(
        self, indices: List[int], train: bool = True
    ) -> torch.utils.data.Subset:
        if train:
            return torch.utils.data.Subset(self.mnist_train, indices)
        return torch.utils.data.Subset(self.mnist_test, indices)

    def load_data(self):
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=(0.5), std=(0.5))]
        )
        self.mnist_train = MNIST(root=self.path, download=True, transform=transform)
        self.mnist_test = MNIST(
            root=self.path, train=False, download=True, transform=transform
        )

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


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        super().__init__()

        # Discriminator will down-sample the input producing a binary output
        self.fc1 = nn.Linear(
            in_features=SHAPE[0] * SHAPE[1] * SHAPE[2], out_features=128
        )
        self.leaky_relu1 = nn.LeakyReLU(negative_slope=0.2)
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.leaky_relu2 = nn.LeakyReLU(negative_slope=0.2)
        self.fc3 = nn.Linear(in_features=64, out_features=32)
        self.leaky_relu3 = nn.LeakyReLU(negative_slope=0.2)
        self.fc4 = nn.Linear(in_features=32, out_features=1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Rehape passed image batch
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)

        # Feed forward
        x = self.fc1(x)
        x = self.leaky_relu1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.leaky_relu2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.leaky_relu3(x)
        x = self.dropout(x)
        logit_out = self.fc4(x)

        return logit_out.flatten()


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # Generator will up-sample the input producing input of size
        # suitable for feeding into discriminator
        self.fc1 = nn.Linear(in_features=Z_DIM, out_features=32)
        self.relu1 = nn.LeakyReLU(negative_slope=0.2)
        self.fc2 = nn.Linear(in_features=32, out_features=64)
        self.relu2 = nn.LeakyReLU(negative_slope=0.2)
        self.fc3 = nn.Linear(in_features=64, out_features=128)
        self.relu3 = nn.LeakyReLU(negative_slope=0.2)
        self.fc4 = nn.Linear(
            in_features=128, out_features=SHAPE[0] * SHAPE[1] * SHAPE[2]
        )
        self.dropout = nn.Dropout(0.3)
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # flatten the input
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)

        # Feed forward
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.dropout(x)
        x = self.fc4(x)
        tanh_out = self.tanh(x)

        # Reshape to image shape
        logit_out = tanh_out.view(-1, SHAPE[0], SHAPE[1], SHAPE[2])

        return logit_out
