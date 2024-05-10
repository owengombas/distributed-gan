import datetime
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
    dataset_is = generated_images[: min(n_samples, generated_images.shape[0])].to(
        dtype=torch.uint8
    )
    metric = InceptionScore(feature=2048)
    metric.update(dataset_is)
    scores = InceptionScore.compute(metric)
    return scores[0].item()


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
    n_samples: int = 5,
    iid: bool = True,
):
    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(weeks=52),
    )
    logging.info(f"Server {rank} initialized")

    log_file: Path = log_folder / f"server_{rank}.logs.json"
    logs: Dict[str, Any] = {}

    generator.to(device, dtype=torch.float32)

    optimizer = torch.optim.Adam(
        generator.parameters(), lr=generator_lr, betas=(0.5, 0.999)
    )

    N = world_size - 1
    X_ns: List[torch.Tensor] = [None] * N
    X_gs: List[torch.Tensor] = [None] * N
    K = N

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2 * N * batch_size, shuffle=True
    )
    list_of_indices = split_dataset(len(dataset), world_size - 1, iid=iid)
    logging.info(f"Server {rank} split the dataset into {len(list_of_indices)} parts")
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
            f"Server {rank} sent indices to worker {i + 1} with shape {indices.shape}"
        )

    generator.train()
    for epoch in range(epochs):
        start_time = time.time()
        logs[epoch] = {
            "epoch": epoch,
            "start_time": start_time,
        }

        real_images: torch.Tensor = next(iter(data_loader))[0]

        for rank in range(1, K + 1):
            t_n = torch.randn((batch_size, z_dim, 1, 1), device=device)
            t_g = torch.randn((batch_size, z_dim, 1, 1), device=device)

            X_n: torch.Tensor = generator(t_n).to(device=device)
            X_g: torch.Tensor = generator(t_g).to(device=device)

            X_gs[rank - 1] = X_g
            X_ns[rank - 1] = X_n

            t = torch.zeros(
                (2, batch_size, *image_shape), dtype=torch.float32, device="cpu"
            )
            t[0] = X_n
            t[1] = X_g
            req = dist.isend(tensor=t.cpu(), dst=rank, tag=1)
            logging.info(f"Server sent data to worker {rank}")
        logs[epoch] = {
            "sent_time": time.time(),
            **logs[epoch],
        }

        feedbacks = torch.zeros(
            (world_size - 1, batch_size, *image_shape),
            dtype=torch.float32,
            requires_grad=True,
            device="cpu",
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
            X_g = X_gs[rank - 1]  # Assuming datasets is indexed starting from 0
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
                        delta_w[j].add_(grad, alpha=1.0 / (batch_size * N))

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
            # Generate some images
            fake_images = torch.cat(X_gs, dim=0).to(device="cpu")

            grid = make_grid(
                fake_images, nrow=4, normalize=True, value_range=(-1, 1), padding=0
            )
            # Convert the grid to a PIL image
            grid_pil = to_pil_image(grid)
            # Create the save directory if it doesn't exist
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            grid_path = (
                Path(save_dir) / f"generated_epoch_{epoch}.png"
            )
            grid_pil.save(grid_path)

            if fake_images.shape[1] < 3:
                fake_images = fake_images.repeat(1, 3, 1, 1)
            if real_images.shape[1] < 3:
                real_images = real_images.repeat(1, 3, 1, 1)

            # normalize the images from 0 to 255
            fake_images = ((fake_images + 1) * 127.5).to(dtype=torch.uint8)
            real_images = ((real_images + 1) * 127.5).to(dtype=torch.uint8)

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
        with open(log_file, "w") as f:
            json.dump(logs, f)

    # Save the generator model
    save_path = Path("saved_models") / "generator.pt"
    torch.save(generator.state_dict(), save_path)
    logging.info(f"Server {rank} saved generator model to {save_path}")

    # Distroy the process group
    dist.destroy_process_group()
    logging.info(f"Server {rank} finished training")
