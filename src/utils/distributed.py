import os
from datetime import timedelta

import torch
import torch.distributed as dist


def setup_dist(timeout_hours: int = 2):
    is_distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ

    if not is_distributed:
        rank = 0
        local_rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return is_distributed, rank, local_rank, world_size, device

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if not torch.cuda.is_available():
        raise RuntimeError("Distributed requested but CUDA is not available (NCCL requires CUDA).")

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available in this build.")

    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=timedelta(hours=timeout_hours),
            device_id=device,
        )

    return is_distributed, rank, local_rank, world_size, device
