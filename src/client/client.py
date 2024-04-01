import argparse
import warnings
import flwr as fl
import numpy as np
import torch

from dataloaders.CifarDataLoader import load_data
from models.CifarCNN import CifarCNN
from clients.CifarClient import CifarClient
PORT = 8080
HOST = "127.0.0.1"
MAX_CLIENTS = 3
BATCH_SIZE = 32
TEST_SIZE = 0.2
N_EPOCHS = 1
MODEL = CifarCNN
CLIENT = CifarClient
DEVICE = torch.device("mps")

warnings.filterwarnings("ignore", category=UserWarning)
parser = argparse.ArgumentParser(description="Flower")
parser.add_argument(
    "--partition-id",
    choices=np.arange(MAX_CLIENTS),
    required=True,
    type=int,
    help="Partition of the dataset divided into 3 iid partitions created artificially.",
)
partition_id = parser.parse_args().partition_id

net = MODEL()
trainloader, testloader = load_data(
    partition_id=partition_id,
    max_partition=MAX_CLIENTS,
    batch_size=BATCH_SIZE,
    test_size=TEST_SIZE,
)
client = CLIENT(
    trainloader=trainloader,
    testloader=testloader,
    net=net,
    epochs=N_EPOCHS,
    device=DEVICE,
)

fl.client.start_client(server_address=f"{HOST}:{PORT}", client=client.to_client())
