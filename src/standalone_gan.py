import argparse
from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from pathlib import Path
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore

from dataloaders.DataPartitioner import DataPartitioner
from dataloaders.CelebaPartitioner import CelebaPartitioner, celeba_shape
from dataloaders.MnistPartitioner import MnistPartitioner, mnist_shape
from dataloaders.Cifar10Partitioner import Cifar10Partitioner, cifar10_shape

from models.CelebaGenerator import CelebaGenerator, celeba_z_dim
from models.MnistGenerator import MnistGenerator, mnist_z_dim
from models.CifarGenerator import CifarGenerator, cifar_z_dim

from models.CelebaDiscriminator import CelebaDiscriminator
from models.MnistDiscriminator import MnistDiscriminator
from models.CifarDiscriminator import CifarDiscriminator

def verify_imports(imports_options: Dict[str, Any], chosen: str) -> None:
    if chosen.lower() not in imports_options:
        raise ValueError(
            f"Option \"{args.dataset}\" not available. Choose from {imports_options.keys()}"
        )

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="cifar")
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--model", type=str, default="cifar")
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--log_images_interval", type=int, default=10)
parser.add_argument("--log_fid_is_interval", type=int, default=500)
parser.add_argument("--n_samples_fid", type=int, default=10)
parser.add_argument("--generator_lr", type=float, default=0.0002)
parser.add_argument("--discriminator_lr", type=float, default=0.0002)
parser.add_argument("--device", type=str, default="cpu")
args = parser.parse_args()

available_datasets: Dict[str, Tuple[DataPartitioner, int]] = {
    "cifar": (Cifar10Partitioner, cifar10_shape),
    "mnist": (MnistPartitioner, mnist_shape),
    "celeba": (CelebaPartitioner, celeba_shape),
}

available_generators: Dict[str, torch.nn.Module] = {
    "cifar": (CifarGenerator, cifar_z_dim),
    "mnist": (MnistGenerator, mnist_z_dim),
    "celeba": (CelebaGenerator, celeba_z_dim),
}

available_discriminators: Dict[str, torch.nn.Module] = {
    "cifar": CifarDiscriminator,
    "mnist": MnistDiscriminator,
    "celeba": CelebaDiscriminator,
}

verify_imports(available_datasets, args.dataset)
verify_imports(available_generators, args.model)
verify_imports(available_discriminators, args.model)

device = torch.device(args.device)
image_shape = available_datasets[args.dataset][1]
nz = available_generators[args.model][1]
dataset: DataPartitioner = available_datasets[args.dataset][0](1, 0)
dataset.load_data()
dataloader = DataLoader(dataset.train_dataset, batch_size=args.batch_size, shuffle=True)

netG = available_generators[args.model][0](image_shape, nz, device)
netD = available_discriminators[args.model](image_shape, device)

criterion = nn.BCELoss()

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

fixed_noise = torch.randn(args.batch_size, nz, 1, 1, device=device)
real_label = 1
fake_label = 0

niter = 25
g_loss = []
d_loss = []

image_output_path: Path = Path("saved_images_standalone")
image_output_path.mkdir(exist_ok=True)

weights_output_path: Path = Path("weights")
weights_output_path.mkdir(exist_ok=True)

def compute_fid_score(real_images: torch.Tensor, fake_images: torch.Tensor, netG: torch.nn.Module) -> float:
    fid = FrechetInceptionDistance(feature=2048)
    fid.update(real_images, real=True)
    fid.update(fake_images, real=False)
    return fid.compute()

def compute_inception_score(fake_images: torch.Tensor, netG: torch.nn.Module) -> float:
    inception = InceptionScore(feature=2048)
    inception.update(fake_images)
    return inception.compute()


for epoch in range(niter):
    for i, data in enumerate(dataloader, 0):
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        netD.zero_grad()
        real_images = data[0].to(device)
        batch_size = real_images.size(0)
        label = torch.full((batch_size,), real_label, device=device, dtype=torch.float32)

        output = netD(real_images)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        # train with fake
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        image_images = netG(noise)
        label.fill_(fake_label)
        output = netD(image_images.detach())
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()

        # (2) Update G network: maximize log(D(G(z)))
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        output = netD(image_images)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        print(f"Epoch {epoch}, Step {i}, Loss D {errD.item()}, Loss G {errG.item()}")

        #save the output
        if i % args.log_images_interval == 0:
            vutils.save_image(real_images, image_output_path/f"real_samples_{epoch}.png", normalize=True)
            image_images = netG(fixed_noise)
            vutils.save_image(image_images.detach(), image_output_path/f"fake_samples_epoch_{epoch}.png", normalize=True)
        
        if i % args.log_fid_is_interval == 0:
            real_images = (real_images+1)*127.5
            fake_images = (image_images+1)*127.5
            real_images = real_images[:args.n_samples_fid].to(device="cpu", dtype=torch.uint8)
            image_images = image_images[:args.n_samples_fid].to(device="cpu", dtype=torch.uint8)
            fid_score = compute_fid_score(real_images, image_images, netG)
            inception_score = compute_inception_score(image_images, netG)
            print(f"Epoch {epoch}, Step {i}, FID {fid_score}, Inception Score {inception_score}")
    
    # Check pointing for every epoch
    torch.save(netG.state_dict(), weights_output_path/f"netG_epoch_{epoch}.pth")
    torch.save(netD.state_dict(), weights_output_path/f"netD_epoch_{epoch}.pth")
