import random

import numpy as np
from torch.utils.data import DataLoader, DistributedSampler

from data.hcp_openacces.dataset import TaskDataset


def get_loader(dataset, batch_size, shuffle, world_size=0, rank=0, num_workers=4):
    if not TaskDataset.fully_downloaded():
        raise RuntimeError(
            "The dataset must be fully downloaded, before calling this function."
        )

    if world_size <= 1:
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
    else:
        data_sampler = DistributedSampler(
            dataset=dataset, num_replicas=world_size, rank=rank, shuffle=shuffle
        )
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            sampler=data_sampler,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    return dataloader


def get_ids(ratios, fraction=1):
    if not TaskDataset.fully_downloaded():
        raise RuntimeError(
            "The dataset must be fully downloaded, before calling this function."
        )

    if not np.isclose(sum(ratios), 1, atol=1e-6):
        raise ValueError("ratios must sum to 1")

    subject_ids = TaskDataset.get_all_subject_ids()
    subset_size = int(len(subject_ids) * fraction)
    shuffled_subset_id = random.sample(subject_ids, subset_size)
    n_train = int(subset_size * ratios[0])
    n_val = int(subset_size * ratios[1])

    return {
        "train": shuffled_subset_id[:n_train],
        "val": shuffled_subset_id[n_train : n_train + n_val],
        "test": shuffled_subset_id[n_train + n_val :],
    }
