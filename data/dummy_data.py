from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, ClassVar, Type

import torch

from helper.class_ import Data_, DataConfig_, Dataset_, Datasets_, DatasetsConfig_


def get_dummy_datasets_config(
    n_classes: int,
    n_samples: int,
    shape: tuple[int, ...],
    split: list[float] = [0.8, 0.1, 0.1],
    dataset_names: list[str] = ["train", "val", "test"],
) -> DummyDatasetsConfig:
    data_config = DummyDataConfig(n_classes=n_classes, n_samples=n_samples, shape=shape)
    return DummyDatasetsConfig(data_config=data_config, split=split, dataset_names=dataset_names)


class DummyData(Data_):
    config: DummyDataConfig

    def __init__(self, config: DummyDataConfig):
        self.config = config

    def size_info(self) -> None:
        print("DummyData size info")

    def download(self) -> None:
        print("DummyData download")


class DummyDatasets(Datasets_):
    config: DummyDatasetsConfig

    class DummyDataset(Dataset_):
        def __init__(self, outer: DummyDatasets, subjects: list[str]):
            super().__init__()
            self.outer = outer
            self.subjects = subjects

        def __len__(self) -> int:
            return len(self.subjects)

        def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
            return torch.randn(self.outer.config.data_config.shape, dtype=torch.float32), random.randrange(
                0, self.outer.config.data_config.n_classes
            )

        def get_class_distribution(self) -> list[int]:
            return self.outer.get_class_distribution(subjects=self.subjects)

    def __init__(self, config: DummyDatasetsConfig):
        datasets: dict[str, Dataset_] = {}
        for name, subjects in config.dataset_ids.items():
            datasets[name] = self.DummyDataset(outer=self, subjects=subjects)

        super().__init__(config=config, datasets=datasets)
        self.data = DummyData(config.data_config)

    def get_dataset(self, name: str) -> Dataset_:
        if name not in self.datasets.keys():
            raise KeyError(f"Dataset `{name}` not found.")

        return self.datasets[name]

    def get_class_distribution(self, subjects: list[str] | None = None) -> list[int]:
        n_classes = self.config.data_config.n_classes
        n_samples = self.config.data_config.n_samples if subjects is None else len(subjects)
        per_class = n_samples // n_classes
        return [per_class] * (n_classes - 1) + [n_samples - per_class * (n_classes - 1)]

    def get_metadata(self) -> list[str]:
        return ["DummyDataset metadata"]

    def size_info(self) -> None:
        self.data.size_info()

    def download(self) -> None:
        self.data.download()


@dataclass
class DummyDataConfig(DataConfig_):
    class_: ClassVar[type[DummyData]] = DummyData
    name: ClassVar[str] = "dummy"

    n_samples: int
    shape: tuple[int, ...]
    n_classes: int

    def __init__(
        self,
        n_samples: int,
        shape: tuple[int, ...],
        n_classes: int,
    ) -> None:
        super().__init__(description="", ids=[str(id_) for id_ in range(n_samples)])
        self.n_samples = n_samples
        self.shape = shape
        self.n_classes = n_classes

    @staticmethod
    def from_dict(dict_: dict[str, Any]) -> DummyDataConfig:
        return DummyDataConfig(**dict_)

    def as_dict(self) -> dict[str, Any]:
        return {"n_samples": self.n_samples, "shape": self.shape, "n_classes": self.n_classes}


@dataclass
class DummyDatasetsConfig(DatasetsConfig_):
    class_: ClassVar[Type[DummyDatasets]] = DummyDatasets

    data_config: DummyDataConfig
    split: list[float]

    def __init__(
        self,
        data_config: DummyDataConfig,
        split: list[float] = [0.8, 0.1, 0.1],
        dataset_names: list[str] = ["train", "val", "test"],
    ) -> None:
        if len(dataset_names) != len(set(dataset_names)):
            raise ValueError("`dataset_names` must contain unique names.")

        if len(split) != len(dataset_names):
            raise ValueError("`split` and `dataset_names` must have the same length.")

        ids = [str(i) for i in range(data_config.n_samples)]

        dataset_ids = {name: ids for name, ids in zip(dataset_names, Datasets_.get_split(ids=ids, split=split))}
        super().__init__(description="", data_config=data_config, dataset_ids=dataset_ids, split=split)

    @classmethod
    def _from_dict(cls, dict_: dict[str, Any]) -> DummyDatasetsConfig:
        data_config = DummyDataConfig.from_dict(
            {
                "n_samples": dict_["data"]["n_samples"],
                "shape": dict_["data"]["shape"],
                "n_classes": dict_["data"]["n_classes"],
            }
        )

        config = object.__new__(DummyDatasetsConfig)
        DatasetsConfig_.__init__(
            config,
            description="",
            data_config=data_config,
            dataset_ids=dict_["datasets"]["dataset_ids"],
            split=dict_["datasets"]["split"],
        )

        return config

    def _as_dict(self) -> dict[str, Any]:
        return {
            "data": self.data_config.as_dict(),
            "datasets": {
                "description": self.description,
                "split": self.split,
                "dataset_ids": self.dataset_ids,
            },
        }

    def get_classes(self, pretty: bool = False) -> list[str]:
        return [f"class_{i}" for i in range(self.data_config.n_classes)]

    def get_dataset_size(self, name: str) -> int:
        return len(self.dataset_ids[name])
