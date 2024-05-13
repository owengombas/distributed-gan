from typing import List, Tuple

import torch
import torch.utils.data
from torchvision.datasets import CelebA
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image

from dataloaders.DataPartitioner import DataPartitioner, _get_partition

celeba_shape = (3, 64, 64)

class CelebaPartitioner(DataPartitioner):
    """
    Partition CelebA dataset
    """

    def __init__(self, world_size: int, rank: int, path: str = "data/celeba"):
        self.world_size = world_size
        self.rank = rank
        self.celeba_train = None
        self.celeba_test = None
        self.path = path

    def load_data(self):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                transforms.Resize((64, 64)),
            ]
        )
        self.celeba_train = CelebA(root=self.path, download=False, transform=transform)
        self.celeba_test = CelebA(root=self.path, split="test", download=False, transform=transform)

    def get_subset_from_indices(self, indices: List[int], train: bool = True) -> torch.utils.data.Subset:
        if train:
            return torch.utils.data.Subset(self.celeba_train, indices)
        return torch.utils.data.Subset(self.celeba_test, indices)

    def get_train_partition(
        self, partition_id: int
    ) -> Tuple[torch.utils.data.Subset, int, int]:
        return _get_partition(self.world_size, partition_id, self.celeba_train)

    def shuffle(self):
        self.celeba_train = torch.utils.data.Subset(
            self.celeba_train, torch.randperm(len(self.celeba_train))
        )
        self.celeba_test = torch.utils.data.Subset(
            self.celeba_test, torch.randperm(len(self.celeba_test))
        )

    def get_test_partition(
        self, partition_id: int
    ) -> Tuple[torch.utils.data.Subset, int, int]:
        return _get_partition(self.world_size, partition_id, self.celeba_test)

    @property
    def train_dataset(self) -> torch.utils.data.Dataset:
        return self.celeba_train

    @property
    def test_dataset(self) -> torch.utils.data.Dataset:
        return self.celeba_test
