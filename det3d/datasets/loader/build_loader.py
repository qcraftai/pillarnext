
from .collate import collate
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import torch.distributed as dist


def build_dataloader(dataset, batch_size, num_workers,  shuffle, pin_memory = False):
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        sampler = DistributedSampler(dataset, num_replicas = world_size, rank = rank, shuffle = shuffle)
    else:
        sampler = None


    # TODO change pin_memory
    data_loader = DataLoader(
        dataset,
        batch_size = batch_size,
        sampler = sampler,
        shuffle=(sampler is None and shuffle),
        num_workers= num_workers,
        collate_fn = collate,
        pin_memory = pin_memory,
    )

    return data_loader
