from dataloaders.DataPartitioner import DataPartitioner
from torchvision.datasets import CelebA
import torch
import torch.utils.data

class CelebAPartitioner(DataPartitioner):
    def __init__(self, world_size: int, rank: int):
        self._world_size = world_size
        self._rank = rank
        self._celeba = None

    def load_data(self):
        self._celeba = CelebA(root="data", download=True)

    def get_partition(self, id: int):
        length = len(self._celeba) // self._world_size
        start = id * length
        end = start + length
        return torch.utils.data.Subset(self._celeba, range(start, end))
