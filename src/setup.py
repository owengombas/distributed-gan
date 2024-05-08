# from torchmetrics.image.fid import FrechetInceptionDistance
# fid = FrechetInceptionDistance(feature=64)

# from dataloaders.CelebaPartitioner import CelebaPartitioner
# d = CelebaPartitioner(world_size=1, rank=0)
# d.load_data()

from dataloaders.Cifar10Partitioner import Cifar10Partitioner
d = Cifar10Partitioner(world_size=1, rank=0)
d.load_data()

from dataloaders.MnistPartitioner import MnistPartitioner
d = MnistPartitioner(world_size=1, rank=0)
d.load_data()

