from typing import Dict, List, Tuple
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import flwr as fl
import numpy as np
from tqdm import tqdm


class CifarClient(fl.client.NumPyClient):
    def __init__(self, trainloader: DataLoader, testloader: DataLoader, net: nn.Module, epochs: int = 1, device: torch.device = torch.device("cpu")):
        self.trainloader: DataLoader = trainloader
        self.testloader: DataLoader = testloader
        self.net: nn.Module = net
        self.epochs: int = epochs
        self.device = device
        self.net.to(self.device)

    def get_parameters(self, config: Dict[str, str]) -> List[np.ndarray]:
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(
        self, parameters: List[np.ndarray]
    ) -> None:
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(
        self, parameters: List[np.array], config: Dict[str, str]
    ) -> Tuple[List[np.array], int, Dict[str, float]]:
        self.set_parameters(parameters)
        self.train()
        return self.get_parameters(config={}), len(self.trainloader), {}

    def evaluate(self, parameters, config: Dict[str, str]) -> Tuple[float, int, Dict[str, float]]:
        self.set_parameters(parameters)
        loss, accuracy = self.test()
        return float(loss), len(self.testloader), {"accuracy": float(accuracy)}

    def train(self):
        """Train the model on the training set."""
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9)
        for _ in range(self.epochs):
            for batch in tqdm(self.trainloader, "Training"):
                images = batch["img"]
                labels = batch["label"]
                optimizer.zero_grad()
                criterion(self.net(images.to(self.device)), labels.to(self.device)).backward()
                optimizer.step()

    def test(self):
        """Validate the model on the test set."""
        criterion = torch.nn.CrossEntropyLoss()
        correct, loss = 0, 0.0
        with torch.no_grad():
            for batch in tqdm(self.testloader, "Testing"):
                images = batch["img"].to(self.device)
                labels = batch["label"].to(self.device)
                outputs = self.net(images)
                loss += criterion(outputs, labels).item()
                correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        accuracy = correct / len(self.testloader.dataset)
        return loss, accuracy
