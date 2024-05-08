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
from torchmetrics.image.inception import InceptionScore
import threading


mp.set_start_method("spawn", force=True)
save_dir: Path = Path("saved_images")


def compute_fid_score(
    fake_images: torch.Tensor,
    real_images: torch.Tensor,
    n_samples: int = torch.inf,
) -> float:
    # Compute Frechet Inception Distance
    fid = FrechetInceptionDistance(feature_dim=2048)
    # shuffle the images
    fake_images = fake_images[torch.randperm(len(fake_images))]
    fake_images_fid: torch.Tensor = fake_images[: min(n_samples, fake_images.shape[0])]
    if fake_images_fid.shape[1] < 3:
        fake_images_fid = fake_images_fid.repeat(1, 3, 1, 1)
    # normalize the images to [0, 1]
    fake_images_fid = (fake_images_fid + 1) / 2
    logging.info(f"{fake_images_fid.shape} fake images")
    fid.update(fake_images_fid, is_real=False)

    # Pick len(all_images) from dataset
    real_images_fid: torch.Tensor = real_images[: min(n_samples, fake_images.shape[0])]
    if real_images_fid.shape[1] < 3:
        real_images_fid = real_images_fid.repeat(1, 3, 1, 1)
    # normalize the images to [0, 1]
    real_images_fid = (real_images_fid + 1) / 2
    logging.info(f"{real_images_fid.shape} real images")
    fid.update(real_images_fid, is_real=True)

    fid_score = fid.compute()
    logging.info(f"Server computed FID score: {fid_score}")

    return fid_score


def compute_inception_score(
    generated_images: torch.Tensor,
) -> float:
    metric = InceptionScore()
    metric.update(generated_images.to(dtype=torch.uint8))
    result = metric.compute()
    logging.info(f"Server computed Inception Score: {result}")
    return result

def generated_batch_sender(
    rank: int,
    generator: torch.nn.Module,
    batch_size: int,
    device: int,
    image_shape: Tuple[int, int, int],
    z_dim: int,
    datasets: List[List[torch.Tensor]],
):
    while True:
        t_n = torch.randn((batch_size, z_dim, 1, 1), device=device)
        t_g = torch.randn((batch_size, z_dim, 1, 1), device=device)

        X_n: torch.Tensor = generator(t_n).to(device=device)
        X_g: torch.Tensor = generator(t_g).to(device=device)
        
        t = torch.zeros(
            (2, batch_size, *image_shape), dtype=torch.float32, device=device
        )
        t[0] = X_n
        t[1] = X_g

        datasets[rank - 1].append(X_n)
        datasets[rank].append(X_g)

        dist.send(tensor=t.clone().cpu(), dst=rank, tag=1)
        logging.info(f"Server sent generated batch to worker {rank}")

def feedback_receiver(rank: int, feedbacks: List[List[torch.Tensor]], batch_size: int, image_shape: Tuple[int, int, int], device: torch.device, current_iteration_count: List[int]):
    while True:
        feedback = torch.zeros((batch_size, *image_shape), dtype=torch.float32, device="cpu")
        dist.recv(tensor=feedback, src=rank, tag=3)
        feedbacks[rank - 1].append(feedback.to(device=device))
        current_iteration_count[0] += 1
        logging.info(f"Server received feedback from worker {rank} with shape {feedback.shape}")


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
        backend=backend,
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(weeks=52),
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
    K = N
    datasets: List[torch.Tensor] = [[]] * (N * 2)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2 * N * batch_size, shuffle=True
    )
    generator.train()

    data_generation_threads: List[threading.Thread] = []
    for rank in range(1, K + 1):
        thread = threading.Thread(
                target=generated_batch_sender,
                args=(
                    rank,
                    generator,
                    batch_size,
                    device,
                    image_shape,
                    z_dim,
                    datasets,
                ),
            )
        data_generation_threads.append(thread)
    for thread in data_generation_threads:
        thread.start()

    feedbacks: List[torch.Tensor] = [[]] * N
    feedback_receiver_threads = []
    current_iteration_count = [0]
    for rank in range(1, K + 1):
        thread = threading.Thread(
            target=feedback_receiver,
            args=(rank, feedbacks, batch_size, image_shape, device, current_iteration_count),
        )
        feedback_receiver_threads.append(thread)
    for thread in feedback_receiver_threads:
        thread.start()
    
    for epoch in range(epochs):
        current_iteration_count[0] = 0

        start_time = time.time()
        logs[epoch] = {
            "epoch": epoch,
            "start_time": start_time,
        }

        logs[epoch] = {
            "sent_time": time.time(),
            **logs[epoch],
        }

        while True:
            if current_iteration_count[0] == N:
                break

        logs[epoch] = {
            "received_time": time.time(),
            **logs[epoch],
        }

        logging.info(f"Server received feedback from all workers")
        # Compute gradient of X_n wrt to w_j (weights of generator)
        delta_w: List[torch.Tensor] = [torch.zeros_like(p, requires_grad=False) for p in generator.parameters()]
        for rank in range(1, world_size):
            feedback = feedbacks[rank - 1].pop(0)
            X_g = datasets[rank - 1].pop(0)
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
            real_images: torch.Tensor = next(iter(data_loader))[0]
    
            # create a process to compute the FID score
            fake_images = generator(torch.randn((n_samples_fid, z_dim, 1, 1), device=device))
            grid = make_grid(
                fake_images, nrow=4, normalize=True, value_range=(-1, 1), padding=0
            )
            # Convert the grid to a PIL image
            grid_pil = to_pil_image(grid)
            # Create the save directory if it doesn't exist
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            grid_path = (
                Path(save_dir) / f"generated_epoch_{epoch}_for_worker_{rank}.png"
            )
            grid_pil.save(grid_path)
            logs[epoch]["fid_calculation_time_start"] = time.time()
            fid_score = compute_fid_score(
                fake_images,
                real_images,
                n_samples_fid,
            )
            logs[epoch]["fid_score"] = fid_score
            logs[epoch]["fid_calculation_time_end"] = time.time()

            # logs[epoch]["is_calculation_time_start"] = time.time()
            # is_score = compute_inception_score(fake_images)
            # logs[epoch]["inception_score"] = is_score
            # logs[epoch]["is_calculation_time_end"] = time.time()

            logs[epoch]["logging_time_end"] = time.time()
            with open(log_file, "w") as f:
                json.dump(logs.copy(), f)

    # Save the generator model
    save_path = Path("saved_models") / "generator.pt"
    torch.save(generator.state_dict(), save_path)
    logging.info(f"Server {rank} saved generator model to {save_path}")

    # Distroy the process group
    dist.destroy_process_group()
    logging.info(f"Server {rank} finished training")
