import argparse
import json
from typing import Dict, Any, Tuple, List
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image
from pathlib import Path
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
import time
import random
import numpy as np
from datasets.DataPartitioner import DataPartitioner
import importlib


def _weights_init(m: nn.Module) -> None:
    """
    Initialize the weights of the network
    :param m: The network
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def _compute_fid_score(
    real_images: torch.Tensor, fake_images: torch.Tensor, device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    """
    Compute the Frechet Inception Distance
    :param real_images: The real images
    :param fake_images: The generated images
    :return: The Frechet Inception Distance
    """
    fid = FrechetInceptionDistance(normalize=True).to(device)
    fid.update(real_images, real=True)
    fid.update(fake_images, real=False)
    return fid.compute()


def _compute_inception_score(fake_images: torch.Tensor, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """
    Compute the inception score
    :param fake_images: The generated images
    :return: The inception score
    """
    inception = InceptionScore(normalize=True, splits=1).to(device)
    inception.update(fake_images)
    return inception.compute()[0]


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="cifar")
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--local_epochs", type=int, default=10)
parser.add_argument("--model", type=str, default="cifar")
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--log_interval", type=int, default=10)
parser.add_argument("--n_samples_fid", type=int, default=10)
parser.add_argument("--generator_lr", type=float, default=0.0002)
parser.add_argument("--discriminator_lr", type=float, default=0.0002)
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--beta_1", type=float, default=0.0)
parser.add_argument("--beta_2", type=float, default=0.999)
args = parser.parse_args()

np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.mps.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True

device = torch.device(args.device)

if __name__ == "__main__":
    # Determine the evaluation device
    evaluation_device = torch.device("cpu")
    if torch.cuda.is_available():
        evaluation_device = torch.device("cuda")
    
    print(f"Running in {device} and evaluating on {evaluation_device}")

    # Dynamically import the dataset module
    dataset_module = importlib.import_module(f"datasets.{args.dataset}")

    # Initialize the data partitioner
    partioner: DataPartitioner = dataset_module.Partitioner
    partioner = partioner(0, 0)
    partioner.load_data()
    dataloader = DataLoader(
        partioner.train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # Initialize the generator and discriminator
    generator: nn.Module = dataset_module.Generator().to(device)
    discriminator: nn.Module = dataset_module.Discriminator().to(device)

    # Initialize the weights of the generator and discriminator
    generator.apply(_weights_init)
    discriminator.apply(_weights_init)

    # Print the summary of the models
    print(discriminator)
    print(generator)

    # Retrieve the image shape and z dimension
    z_dim: int = dataset_module.Z_DIM

    # Retrieve the arguments of the program
    batch_size: int = args.batch_size
    n_samples_fid: int = args.n_samples_fid
    epochs: int = args.epochs
    local_epochs: int = args.local_epochs
    generator_lr: float = args.generator_lr
    discriminator_lr: float = args.discriminator_lr
    beta_1: float = args.beta_1
    beta_2: float = args.beta_2

    # Setup the loss function and optimizers
    criterion = nn.BCELoss()
    optimizer_discriminator = optim.Adam(
        discriminator.parameters(), lr=discriminator_lr, betas=(beta_1, beta_2)
    )
    optimizer_generator = optim.Adam(
        generator.parameters(), lr=generator_lr, betas=(beta_1, beta_2)
    )

    fixed_noise = torch.randn(args.batch_size, z_dim, 1, 1, device=device)
    real_label = torch.ones(args.batch_size, device=device)
    fake_label = torch.zeros(args.batch_size, device=device)
    g_loss: List[float] = []
    d_loss: List[float] = []

    image_output_path: Path = Path("saved_images_standalone")
    weights_output_path: Path = Path("weights")
    logs_output_path: Path = Path("logs")
    logs_filename = f"{args.dataset}.standalone.logs.json"
    logs = []
    for epoch in range(epochs):
        current_logs = {
            "epoch": epoch,
            "start.epoch": time.time(),
            "end.epoch": None,
            "start.epoch_calculation": time.time(),
            "start.discriminator_train": None,
            "end.discriminator_train": None,
            "start.generator_train": None,
            "end.generator_train": None,
            "end.epoch_calculation": None,
            "absolut_step": epoch * local_epochs,
            "mean_d_loss": None,
            "mean_g_loss": None,
            "start.train": time.time(),
            "end.train": None,
            "start.fid": None,
            "end.fid": None,
            "start.is": None,
            "end.is": None,
            "fid": None,
            "is": None,
        }

        real_images = next(iter(dataloader))[0].to(device)
        
        noise = torch.randn(batch_size, z_dim, 1, 1, device=device)
        fake_images: torch.Tensor = generator(noise)

        losses_d = torch.zeros(local_epochs, device=device, dtype=torch.float32)
        losses_g = torch.zeros(local_epochs, device=device, dtype=torch.float32)
        for i in range(local_epochs):
            # (1) Update discriminator network by maximizing log(D(x)) + log(1 - D(G(z)))
            current_logs["start.discriminator_train"] = time.time()
            discriminator.zero_grad()
            output: torch.Tensor = discriminator(real_images)
            errD_real: torch.Tensor = criterion(output, real_label)
            errD_real.backward()

            # Train with fake
            output: torch.Tensor = discriminator(fake_images.detach())
            errD_fake: torch.Tensor = criterion(output, fake_label)
            errD_fake.backward()
            errD: torch.Tensor = errD_real + errD_fake
            losses_d[i] = errD
            optimizer_discriminator.step()
            current_logs["end.discriminator_train"] = time.time()

            current_logs["start.generator_train"] = time.time()
            # (2) Update the generator network by maximizing log(D(G(z)))
            generator.zero_grad()
            output: torch.Tensor = discriminator(fake_images)
            errG: torch.Tensor = criterion(output, real_label)
            losses_g[i] = errG
            errG.backward()
            optimizer_generator.step()
            current_logs["end.generator_train"] = time.time()

        current_logs["mean_d_loss"] = losses_d.mean().item()
        current_logs["mean_g_loss"] = losses_g.mean().item()
        current_logs["end.epoch_calculation"] = time.time()
        current_logs["start.train"] = time.time()

        print(
            f"Epoch {epoch}, Step {i}, Loss D {errD.item()}, Loss G {errG.item()}"
        )

        if epoch % args.log_interval == 0:
            # Normalize the images to [0, 1] range instead of [-1, 1]
            real_images = (real_images + 1) * 0.5
            fake_images = (fake_images + 1) * 0.5

            # If the images are grayscale, repeat the channels to make them RGB
            if fake_images.shape[1] < 3:
                fake_images = fake_images.repeat(1, 3, 1, 1)
            if real_images.shape[1] < 3:
                real_images = real_images.repeat(1, 3, 1, 1)

            # FID and IS are performed on CPU
            real_images = real_images[: args.n_samples_fid].to(device=evaluation_device)
            fake_images = fake_images[: args.n_samples_fid].to(device=evaluation_device)

            # Save the images to check the progress of the generator
            image_output_path.mkdir(parents=True, exist_ok=True)
            grid = make_grid(
                fake_images, nrow=4, normalize=True, value_range=(0, 1), padding=0
            )
            grid_pil = to_pil_image(grid.cpu())
            grid_pil.save(image_output_path / f"fake_samples_{epoch}.png")

            # Compute FID and IS
            current_logs["start.fid"] = time.time()
            fid_score = _compute_fid_score(real_images, fake_images, evaluation_device).item()
            current_logs["end.fid"] = time.time()
            current_logs["fid"] = fid_score

            current_logs["start.is"] = time.time()
            inception_score = _compute_inception_score(fake_images, evaluation_device).item()
            current_logs["end.is"] = time.time()
            current_logs["is"] = inception_score

            print(
                f"Epoch {epoch}, Step {i}, FID {fid_score}, Inception Score {inception_score}"
            )

        logs_output_path.mkdir(parents=True, exist_ok=True)
        current_logs["end.epoch"] = time.time()
        logs.append(current_logs)
        with open(logs_output_path / logs_filename, "w") as f:
            json.dump(logs, f)

        # Check pointing for every epoch
        weights_output_path.mkdir(parents=True, exist_ok=True)
        torch.save(generator.state_dict(), weights_output_path / f"netG_epoch_{epoch}.pth")
        torch.save(discriminator.state_dict(), weights_output_path / f"netD_epoch_{epoch}.pth")
