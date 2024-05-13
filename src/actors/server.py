import datetime
import math
from typing import List, Tuple
import torch.distributed as dist
import torch
from torch.futures import Future
import torch.utils.data
import logging
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image
from pathlib import Path
import multiprocessing as mp
import time
from pathlib import Path
from typing import Dict, Any
import json
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore

torch.manual_seed(42)
save_dir: Path = Path("saved_images")


def compute_fid_score(
    fake_images: torch.Tensor,
    real_images: torch.Tensor,
    n_samples: int = torch.inf,
    device: torch.device = torch.device("cpu"),
) -> float:
    # Compute Frechet Inception Distance
    fid = FrechetInceptionDistance(feature=2048)
    # shuffle the images
    fake_images = fake_images[torch.randperm(len(fake_images))]
    fake_images_fid: torch.Tensor = fake_images[
        : min(n_samples, fake_images.shape[0])
    ].to(device="cpu", dtype=torch.uint8)
    logging.info(f"Computing FID score")
    logging.info(f"{fake_images_fid.shape} fake images")
    fid.update(fake_images_fid, real=False)

    # Pick len(all_images) from dataset
    real_images_fid: torch.Tensor = real_images[
        : min(n_samples, fake_images.shape[0])
    ].to(device="cpu", dtype=torch.uint8)
    logging.info(f"{real_images_fid.shape} real images")
    fid.update(real_images_fid, real=True)

    fid_score = FrechetInceptionDistance.compute(fid)
    logging.info(f"Server computed FID score: {fid_score}")

    return fid_score.item()


def compute_inception_score(
    generated_images: torch.Tensor,
    n_samples: int = torch.inf,
    device: torch.device = torch.device("cpu"),
) -> float:
    logging.info(f"Computing Inception score")
    dataset_is = generated_images[: min(n_samples, generated_images.shape[0])].to(
        dtype=torch.uint8
    )
    metric = InceptionScore(feature=2048)
    metric.update(dataset_is)
    scores = InceptionScore.compute(metric)
    mean_score = scores[0].item()
    logging.info(f"Server computed Inception score: {mean_score}")
    return mean_score


def split_dataset(
    dataset_size: int, world_size: int, iid: bool = False
) -> List[torch.Tensor]:
    """
    Split the dataset into N parts, where N is the number of workers.
    Each worker will get a different part of the dataset.
    :param dataset_size: The size of the dataset
    :param world_size: The number of workers
    :param rank: The rank of the current worker
    :param iid: Whether to shuffle the dataset before splitting
    :return: A list of tensors, each tensor hold the indices of the dataset for each worker
    """
    if iid:
        indices = torch.randperm(dataset_size, device=torch.device("cpu"))
    else:
        indices = torch.arange(dataset_size, device=torch.device("cpu"))
    # Split the dataset into N parts
    split_indices = torch.chunk(indices, world_size)
    return split_indices


def server(
    backend: str,
    i: int,
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
    n_samples: int = 5,
    iid: bool = True,
):
    dist.init_process_group(
        backend=backend,
        rank=i,
        world_size=world_size,
        timeout=datetime.timedelta(weeks=52),
    )
    logging.info(f"Server {i} initialized")

    log_file: Path = log_folder / f"server_{i}.logs.json"
    logs: Dict[str, Any] = {}

    generator.to(device, dtype=torch.float32)

    optimizer = torch.optim.Adam(
        generator.parameters(), lr=generator_lr, betas=(0, 0.999)
    )

    N = world_size - 1
    K = math.floor(math.log2(N))
    logging.info(f"Server {i} has {N} workers and K={K}")

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2 * N * batch_size, shuffle=True
    )

    real_images: torch.Tensor = next(iter(data_loader))[0].cpu()
    if real_images.shape[1] < 3:
        real_images = real_images.repeat(1, 3, 1, 1)
    real_images = ((real_images + 1) * 127.5).to(dtype=torch.uint8)

    grid_real = make_grid(
        real_images.to(dtype=torch.float32), nrow=4, normalize=True, value_range=(0, 255), padding=0
    )
    # Convert the grid to a PIL image
    grid_pil = to_pil_image(grid_real)
    # Create the save directory if it doesn't exist
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    grid_path = Path(save_dir) / f"real_images.png"
    grid_pil.save(grid_path)

    list_of_indices = split_dataset(len(dataset), world_size - 1, iid=iid)
    logging.info(f"Server {i} split the dataset into {len(list_of_indices)} parts")
    for i, indices in enumerate(list_of_indices):
        dist.send(
            tensor=torch.tensor(
                len(indices), device=torch.device("cpu"), dtype=torch.int
            ),
            dst=i + 1,
            tag=4,
        )
        dist.send(tensor=indices, dst=i + 1, tag=4)
        logging.info(
            f"Server {i} sent indices to worker {i + 1} with shape {indices.shape}"
        )
    
    weights_path = Path("weights")

    generator.train()
    feedbacks = torch.zeros((N, batch_size, *image_shape), device="cpu", requires_grad=True, dtype=torch.float32)
    for epoch in range(epochs):
        logging.info(f"Server {i} starting epoch {epoch}")
        start_time = time.time()
        logs[epoch] = {
            "epoch": epoch,
            "start_time": start_time,
        }

        seed = torch.randn((2 * K * batch_size, z_dim, 1, 1), device=device)
        X: torch.tensor = generator(seed).to(device=device)
        X_gs = [X[(k + 1) * batch_size : (k + 2) * batch_size] for k in range(K)]
        reqs = [None] * N

        logs[epoch] = {
            "sent_time": time.time(),
            **logs[epoch],
        }

        feedbacks = feedbacks.to(device="cpu")
        for i in range(N):
            logging.info(f"Server receiving feedback from worker {i+1}")
            reqs[i] = dist.irecv(tensor=feedbacks[i], src=i+1, tag=3)

            k = i % K
            t_n: torch.Tensor = X[k * batch_size : (k + 2) * batch_size]
            logging.info(
                f"Server sending generated data {k} with shape {t_n.shape} to worker {i+1}"
            )
            dist.isend(tensor=t_n.cpu(), dst=i+1, tag=1)
            logging.info(f"Server sent data to worker {i}")
        for i in range(N):
            reqs[i].wait()
        feedbacks = feedbacks.to(device=device)
        
        logging.info(f"Server received feedback from all workers")

        logs[epoch] = {
            "received_time": time.time(),
            **logs[epoch],
        }

        # Precompute some constants
        inverse_batch_size_N = 1.0 / (batch_size * N)

        # Aggregate gradients for each batch and parameter
        grads_sum = [
            torch.zeros_like(p, requires_grad=False, device=device)
            for p in generator.parameters()
        ]

        # Pre-compute gradients for all feedbacks in a batch
        for i in range(N):
            k = i % K
            X_g = X_gs[k]

            # Compute gradients for the entire batch at once if possible
            # Flatten X_g and feedback to match the batch dimensions if necessary
            batched_X_g = torch.cat([x_i.unsqueeze(0) for x_i in X_g], dim=0)
            batched_feedback = torch.cat([e_i.unsqueeze(0) for e_i in feedbacks[i]], dim=0)

            # Calculate gradients for the whole batch
            batch_grads = torch.autograd.grad(
                outputs=batched_X_g,
                inputs=generator.parameters(),
                grad_outputs=batched_feedback,
                retain_graph=True,
                allow_unused=True
            )

            # Aggregate the gradients
            for j, grad in enumerate(batch_grads):
                if grad is not None:
                    grads_sum[j] += grad

        # Average the gradients and update delta_w
        delta_w = [
            g * inverse_batch_size_N for g in grads_sum
        ]

        logging.info(f"Server aggregated the gradients from all workers")

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
            fake_images = X.detach().cpu()
            if fake_images.shape[1] < 3:
                fake_images = fake_images.repeat(1, 3, 1, 1)
            # normalize the images from 0 to 255
            fake_images = ((fake_images + 1) * 127.5).to(dtype=torch.uint8)
            logging.info(f"Server {fake_images.shape} generated images")

            grid_fake = make_grid(
                fake_images.to(dtype=torch.float32), nrow=4, normalize=True, value_range=(0, 255), padding=0
            )
            # Convert the grid to a PIL image
            grid_pil = to_pil_image(grid_fake)
            # Create the save directory if it doesn't exist
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            grid_path = Path(save_dir) / f"generated_epoch_{epoch}.png"
            grid_pil.save(grid_path)

            logs[epoch]["is_calculation_time_start"] = time.time()
            is_score = compute_inception_score(fake_images, n_samples, device="cpu")
            logs[epoch]["inception_score"] = is_score
            logs[epoch]["is_calculation_time_end"] = time.time()
            logs[epoch]["fid_calculation_time_start"] = time.time()
            fid_score = compute_fid_score(
                fake_images, real_images, n_samples, device=device
            )
            logs[epoch]["fid_score"] = fid_score
            logs[epoch]["fid_calculation_time_end"] = time.time()

            logs[epoch]["logging_time_end"] = time.time()

            weights_path.mkdir(parents=True, exist_ok=True)
            torch.save(generator.state_dict(), weights_path / f"generator_{epoch}.pt")

        with open(log_file, "w") as f:
            json.dump(logs, f)

    # Save the generator model
    save_path = Path("saved_models") / "generator.pt"
    torch.save(generator.state_dict(), save_path)
    logging.info(f"Server {i} saved generator model to {save_path}")

    # Distroy the process group
    dist.destroy_process_group()
    logging.info(f"Server {i} finished training")
