import datetime
from typing import List, Tuple
import torch.distributed as dist
import torch
import torch.nn as nn
from torch.futures import Future
import torch.utils.data
import logging
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image
from torchvision import transforms
from pathlib import Path
import os
from torcheval.metrics import FrechetInceptionDistance
import multiprocessing as mp
import time
from pathlib import Path
from typing import Dict, Any
import json

mp.set_start_method("spawn", force=True)
save_dir: Path = Path("saved_images")


def compute_fid_score(
    epoch: int,
    fake_images: torch.Tensor,
    real_images: torch.Tensor,
    current_logs: Dict[str, Any],
    file: Path,
    n_samples: int = torch.inf,
) -> float:
    # Compute Frechet Inception Distance
    fid = FrechetInceptionDistance(feature_dim=2048)
    # shuffle the images
    fake_images = fake_images[torch.randperm(len(fake_images))]
    fake_images_fid: torch.Tensor = fake_images[
        : min(n_samples, fake_images.shape[0])
    ]
    if fake_images_fid.shape[1] < 3:
        fake_images_fid = fake_images_fid.repeat(1, 3, 1, 1)
    # normalize the images to [0, 1]
    fake_images_fid = (fake_images_fid + 1) / 2
    logging.info(f"{fake_images_fid.shape} fake images")
    fid.update(fake_images_fid, is_real=False)

    # Pick len(all_images) from dataset
    real_images_fid: torch.Tensor = real_images[
        : min(n_samples, fake_images.shape[0])
    ]
    if real_images_fid.shape[1] < 3:
        real_images_fid = real_images_fid.repeat(1, 3, 1, 1)
    # normalize the images to [0, 1]
    real_images_fid = (real_images_fid + 1) / 2
    logging.info(f"{real_images_fid.shape} real images")
    fid.update(real_images_fid, is_real=True)

    logging.info(f"Server computing FID score")
    fid_score = fid.compute()
    logging.info(f"Server computed FID score: {fid_score}")

    epoch_dict = current_logs[epoch]
    epoch_dict["fid"] = fid_score.item()
    epoch_dict["logging_time_end"] = time.time()
    current_logs[epoch] = {**epoch_dict}
    with open(file, "w") as f:
        json.dump(current_logs.copy(), f)


def server(
    backend: str,
    rank: int,
    generator_lr: float,
    world_size: int,
    batch_size: int,
    epochs: int,
    log_interval: int,
    generator: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    z_dim: int,
    log_folder: Path,
    image_shape: Tuple[int, int, int],
    device: torch.device = torch.device("cpu"),
    n_samples_fid: int = 5,
):
    dist.init_process_group(
        backend=backend, rank=rank, world_size=world_size, timeout=datetime.timedelta(weeks=52)
    )
    logging.info(f"Server {rank} initialized")

    log_file: Path = log_folder / f"server_{rank}.logs.json"
    manager = mp.Manager()
    logs: Dict[str, Any] = manager.dict()

    generator.to(device, dtype=torch.float32)

    optimizer = torch.optim.Adam(
        generator.parameters(), lr=generator_lr, betas=(0.5, 0.999)
    )

    N = world_size - 1
    datasets: List[torch.Tensor] = [None] * N
    K = N

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2 * N * batch_size, shuffle=True
    )
    generator.train()
    for epoch in range(epochs):
        start_time = time.time()
        logs[epoch] = {
            "epoch": epoch,
            "start_time": start_time,
        }

        reqs: List[Future] = [None] * N
        fake_images: torch.Tensor = torch.zeros(
            (2 * N * batch_size, *image_shape), dtype=torch.float32, device=device
        )
        real_images: torch.Tensor = next(iter(data_loader))[0]

        for rank in range(1, K + 1):
            t_n = torch.randn((batch_size, z_dim, 1, 1), device=device)
            t_g = torch.randn((batch_size, z_dim, 1, 1), device=device)

            X_n: torch.Tensor = generator(t_n)
            X_g: torch.Tensor = generator(t_g)

            X_n_images = X_n.to(device=device)
            X_g_images = X_g.to(device=device)

            t = torch.zeros((2, batch_size, *image_shape), dtype=torch.float32, device=device)
            t[0] = X_n
            t[1] = X_g

            datasets[rank - 1] = X_n

            req = dist.isend(tensor=t.clone().cpu(), dst=rank, tag=1)
            reqs[rank - 1] = req
            logging.info(f"Server sent data to worker {rank}")
        for req in reqs:
            req.wait()
        logs[epoch] = {
            "sent_time": time.time(),
            **logs[epoch],
        }

        feedbacks = torch.zeros(
            (world_size - 1, batch_size, *image_shape),
            dtype=torch.float32,
            requires_grad=True,
        )
        reqs = [None] * (world_size - 1)
        for rank in range(1, world_size):
            logging.info(
                f"Server waiting for feedback from worker {rank} with shape {feedbacks[rank-1].shape}"
            )
            req = dist.irecv(tensor=feedbacks[rank - 1], src=rank, tag=3)
            reqs[rank - 1] = req
        for i, req in enumerate(reqs):
            req.wait()

        logs[epoch] = {
            "received_time": time.time(),
            **logs[epoch],
        }

        feedbacks = feedbacks.to(device=device)
        # Compute gradient of X_n wrt to w_j (weights of generator)
        delta_w: List[torch.Tensor] = [
            torch.zeros_like(p, requires_grad=False) for p in generator.parameters()
        ]
        for rank in range(1, world_size):
            feedback = feedbacks[rank - 1]
            X_g = datasets[rank - 1]  # Assuming datasets is indexed starting from 0
            for x_i, e_i in zip(X_g, feedback):
                # Compute gradients for each parameter based on the feedback
                grads = torch.autograd.grad(
                    outputs=x_i,  # Assuming generator takes x_i as input to generate data
                    inputs=generator.parameters(),
                    grad_outputs=e_i,
                    retain_graph=True,
                    allow_unused=True,  # In case some parameters don't affect outputs
                )
                # Update delta_w with the new gradients
                for j, grad in enumerate(grads):
                    if grad is not None:
                        delta_w[j].add_(grad, alpha=batch_size * N)

        # Apply the aggregated gradients to the generator
        optimizer.zero_grad()
        for i, p in enumerate(generator.parameters()):
            if delta_w[i] is not None:
                p.grad = delta_w[i].detach()  # Ensure the grad doesn't carry history
        optimizer.step()
        end_time = time.time()
        elapsed_time = end_time - start_time
        logs[epoch] = {
            "end_time": end_time,
            "elapsed_time": elapsed_time,
            "logging_time_end": end_time,
            **logs[epoch],
        }
        logging.info(f"Server applied the gradients, took {elapsed_time} seconds")

        if epoch % log_interval == 0 or epoch == epochs - 1:
            # create a process to compute the FID score
            fake_images[(rank - 1) * 2 * batch_size : rank * 2 * batch_size] = (
                torch.cat((X_n_images, X_g_images), dim=0).cpu()
            )

            images = torch.cat((X_n_images, X_g_images), dim=0)
            grid = make_grid(
                images, nrow=4, normalize=True, value_range=(-1, 1), padding=0
            )
            # Convert the grid to a PIL image
            grid_pil = to_pil_image(grid.cpu())
            # Create the save directory if it doesn't exist
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            grid_path = (
                Path(save_dir) / f"generated_epoch_{epoch}_for_worker_{rank}.png"
            )
            grid_pil.save(grid_path)

            p = mp.Process(
                target=compute_fid_score,
                args=(
                    epoch,
                    fake_images.detach(),
                    real_images.detach(),
                    logs,
                    log_file,
                    n_samples_fid,
                ),
            )
            p.start()

    # Save the generator model
    save_path = Path("saved_models") / "generator.pt"
    torch.save(generator.state_dict(), save_path)
    logging.info(f"Server {rank} saved generator model to {save_path}")

    # Distroy the process group
    dist.destroy_process_group()
    logging.info(f"Server {rank} finished training")
