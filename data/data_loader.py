from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch.utils.data import DataLoader, DistributedSampler

from helper import system
from helper.class_ import Dataset_
from helper.validator import validator


@validator.constraints("world_size", "x >= 0")
@validator.constraints("rank", "x >= 0")
@validator.constraints("num_workers", "x >= 0")
def get_loader(
    config: DataLoaderConfig,
    world_size: int = 0,
    rank: int = 0,
    num_workers: int | None = None,
    prefetch_factor: int | None = None,
    seed: int | None = None,
) -> DataLoader:
    num_workers_ = num_workers if num_workers is not None else system.get_num_workers()
    prefetch_factor_ = prefetch_factor if prefetch_factor is not None else system.get_prefetch_factor()
    seed_ = seed if seed is not None else system.get_seed()

    if world_size <= 1:
        loader = DataLoader(
            dataset=config.dataset,
            batch_size=config.batch_size,
            shuffle=config.shuffle,
            generator=torch.Generator().manual_seed(seed_),
            num_workers=num_workers_,
            pin_memory=world_size > 0,
            prefetch_factor=prefetch_factor_,
            persistent_workers=True,
        )
    else:
        sampler: DistributedSampler = DistributedSampler(
            dataset=config.dataset, num_replicas=world_size, rank=rank, shuffle=config.shuffle, seed=seed_
        )
        loader = DataLoader(
            dataset=config.dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=num_workers_,
            pin_memory=True,
            prefetch_factor=prefetch_factor_,
            persistent_workers=True,
        )

    return loader


@dataclass(init=False)
class DataLoadersConfig:
    batch_size: int
    shuffle_first: bool

    @validator.constraints("batch_size", "x > 0")
    def __init__(self, batch_size: int, shuffle_first: bool):
        self.batch_size = batch_size
        self.shuffle_first = shuffle_first

    def to_configs(self, datasets: list[Dataset_]) -> list[DataLoaderConfig]:
        configs = [DataLoaderConfig(dataset=dataset, batch_size=self.batch_size, shuffle=False) for dataset in datasets]

        if self.shuffle_first:
            configs[0].shuffle = True

        return configs

    @staticmethod
    def from_dict(dict_: dict[str, Any]) -> DataLoadersConfig:
        return DataLoadersConfig(**dict_)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(init=False)
class DataLoaderConfig:
    dataset: Dataset_
    batch_size: int
    shuffle: bool

    @validator.constraints("batch_size", "x > 0")
    def __init__(self, dataset: Dataset_, batch_size: int, shuffle: bool):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
