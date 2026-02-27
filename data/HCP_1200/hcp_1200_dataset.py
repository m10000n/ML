from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Literal, Type, cast

import safetensors.torch as st
import torch

from data import data_helper
from data.HCP_1200.hcp_1200_data import HCP1200Data, HCP1200DataConfig
from helper.class_ import Dataset_, Datasets_, DatasetsConfig_
from helper.validator import validator

_CLASS_TYPES = Literal["task", "subtask"]
_NORMALIZATIONS = Literal[
    "mean_std_window",
    "mean_std_file",
    "mean_std_voxel_window",
    "mean_std_voxel_file",
    "min_max_window",
    "min_max_file",
    "min_max_voxel_window",
    "min_max_voxel_file",
    "none",
]


class HCP1200Datasets(Datasets_):
    config: HCP1200DatasetsConfig

    class HCP1200Dataset(Dataset_):
        def __init__(self, outer: HCP1200Datasets, subjects: list[str], is_train: bool):
            super().__init__()
            self.outer = outer
            self.subjects = subjects
            self.is_train = is_train
            self.samples: list[tuple[Path, tuple[str, str], str]] | None = None
            self.class_distribution: dict[str, float] | None = None

        def __len__(self) -> int:
            if not self.outer.data.is_downloaded():
                raise ValueError("The dataset must be downloaded before calling `__len__`.")

            if self.samples is None:
                self.samples = self.outer.data.get_data_label_id(self.subjects)

            return len(self.samples)

        def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
            if not self.outer.data.is_downloaded():
                raise ValueError("The dataset must be downloaded before calling `__getitem__`.")

            if self.samples is None:
                self.samples = self.outer.data.get_data_label_id(self.subjects)

            file_path, label, id = cast(list, self.samples)[idx]

            try:
                data = st.load_file(filename=file_path)
            except Exception as e:
                raise RuntimeError(
                    f"Error loading file {file_path}. Deleting and re-downloading this file might fix the issue."
                ) from e

            mri = data["tensor"].to(torch.float32)

            if self.outer.config.class_type == "task":
                label_ = label[0]
            elif self.outer.config.class_type == "subtask":
                label_ = label[1]
            else:
                assert False, "Invalid class type."

            if self.is_train and self.outer.config.time_augmentation:
                mri = data_helper.get_window(x=mri, window_width=self.outer.config.window_width)
            else:
                mri = data_helper.get_window(x=mri, window_width=self.outer.config.window_width, start_idx=0)

            spatial_augmentation = self.outer.config.spatial_augmentation
            if self.is_train and spatial_augmentation is not None:
                mri = data_helper.random_transform(
                    mri,
                    angle_max_deg=spatial_augmentation["angle_max_deg"],
                    trans_max_voxel=spatial_augmentation["trans_max_voxel"],
                    scale_max=spatial_augmentation["scale_max"],
                )

            normalization = self.outer.config.normalization
            eps = 1e-8

            mean: float | torch.Tensor
            std: float | torch.Tensor

            if normalization in ["mean_std_window", "mean_std_file", "mean_std_voxel_window", "mean_std_voxel_file"]:
                if normalization == "mean_std_window":
                    mean = float(mri.mean())
                    std = float(mri.std())
                elif normalization == "mean_std_file":
                    mean = float(data["mean"])
                    std = float(data["std"])
                elif normalization == "mean_std_voxel_window":
                    mean = mri.mean(dim=0, keepdim=True)
                    std = mri.std(dim=0, keepdim=True) + eps
                else:
                    mean = data["mean_voxel"].unsqueeze(0)
                    std = data["std_voxel"].unsqueeze(0) + eps

                mri = (mri - mean) / std
            elif normalization in ["min_max_window", "min_max_file", "min_max_voxel_window", "min_max_voxel_file"]:
                min: float | torch.Tensor
                max: float | torch.Tensor
                range: float | torch.Tensor
                if normalization == "min_max_window":
                    min = 0.0
                    max = float(mri.max())
                    range = max
                elif normalization == "min_max_file":
                    min = 0.0
                    max = float(data["max"])
                    range = max
                elif normalization == "min_max_voxel_window":
                    min = mri.amin(dim=0, keepdim=True)
                    max = mri.amax(dim=0, keepdim=True)
                    range = max - min + eps
                else:
                    min = data["min_voxel"].unsqueeze(0)
                    max = data["max_voxel"].unsqueeze(0)
                    range = max - min + eps

                mri = (mri - min) / range
            elif normalization == "none":
                pass
            else:
                assert False, "Invalid normalization."

            return mri, self.outer.encode_class(label_), id

        def get_class_distribution(self) -> list[int]:
            return self.outer.get_class_distribution(self.subjects)

    def __init__(
        self,
        config: HCP1200DatasetsConfig,
    ):
        datasets: dict[str, Dataset_] = {}
        for idx, (name, subjects) in enumerate(config.dataset_ids.items()):
            is_train = idx == 0
            datasets[name] = self.HCP1200Dataset(outer=self, subjects=subjects, is_train=is_train)

        super().__init__(config=config, datasets=datasets)

        self.data = HCP1200Data(config.data_config)
        self.classes = self.config.get_classes()
        self._label_encoding_map = self._get_label_encoding_map()
        self._label_decoding_map = self._get_label_decoding_map()

    def get_dataset(self, name: str) -> Dataset_:
        if name not in self.datasets.keys():
            raise KeyError(f"Dataset `{name}` not found.")

        return self.datasets[name]

    def get_class_distribution(self, subjects: list[str] | None = None) -> list[int]:
        class_distribution = self.data.get_class_distribution(subjects)[self.config.class_type]

        class_distribution_ = [-1] * len(self.classes)

        for class_ in class_distribution:
            class_distribution_[self.encode_class(class_)] = class_distribution[class_]

        return class_distribution_

    def get_metadata(self) -> list[str]:
        return self.data.get_metadata()

    def size_info(self) -> None:
        self.data.size_info()

    def download(self) -> None:
        self.data.download()

    def encode_class(self, class_: str) -> int:
        return self._label_encoding_map[class_]

    def decode_label(self, idx: int) -> str:
        return self._label_decoding_map[idx]

    def _get_label_encoding_map(self) -> dict[str, int]:
        return {label: idx for idx, label in enumerate(self.classes)}

    def _get_label_decoding_map(self) -> dict[int, str]:
        return {idx: label for idx, label in enumerate(self.classes)}


@dataclass(init=False)
class HCP1200DatasetsConfig(DatasetsConfig_):
    class_: ClassVar[Type[HCP1200Datasets]] = HCP1200Datasets

    data_config: HCP1200DataConfig
    normalization: _NORMALIZATIONS
    window_width: int
    class_type: _CLASS_TYPES
    time_augmentation: bool  #
    spatial_augmentation: dict[Literal["angle_max_deg", "trans_max_voxel", "scale_max"], float] | None
    split: list[float]

    @validator.constraints("window_width", "x > 0")
    def __init__(
        self,
        description: str,
        data_config: HCP1200DataConfig,
        split: list[float],
        dataset_names: list[str],
        normalization: _NORMALIZATIONS,
        window_width: int = 27,
        class_type: _CLASS_TYPES = "task",
        time_augmentation: bool = False,
        spatial_augmentation: dict[Literal["angle_max_deg", "trans_max_voxel", "scale_max"], float] | None = None,
    ):
        if len(dataset_names) != len(set(dataset_names)):
            raise ValueError("`dataset_names` must contain unique names.")

        if len(split) != len(dataset_names):
            raise ValueError("`split` and `dataset_names` must have the same length.")

        if data_config.mode == "unprocessed":
            if normalization in [
                "mean_std_voxel_window",
                "mean_std_voxel_file",
                "min_max_voxel_window",
                "min_max_voxel_file",
            ]:
                raise ValueError("Voxel-wise normalization does not work for unprocessed data.")

            if spatial_augmentation is not None and any(value < 0 for value in spatial_augmentation.values()):
                raise ValueError("All values in `spatial_augmentation` must be >= 0.")
        else:
            if spatial_augmentation is not None:
                raise ValueError("Spatial augmentation makes no sense for preprocessed data.")

        if data_config.min_volumes and window_width > data_config.min_volumes:
            raise ValueError(
                f"`window_width` ({window_width}) must be <= `dataset.min_volumes` ({data_config.min_volumes})."
            )

        dataset_subjects = {
            name: ids for name, ids in zip(dataset_names, Datasets_.get_split(ids=data_config.ids, split=split))
        }

        super().__init__(description=description, data_config=data_config, dataset_ids=dataset_subjects, split=split)

        self.data_config = cast(HCP1200DataConfig, data_config)
        self.normalization = normalization
        self.window_width = window_width
        self.class_type = class_type
        self.time_augmentation = time_augmentation
        self.spatial_augmentation = spatial_augmentation

    @classmethod
    def _from_dict(cls, dict_: dict[str, Any]) -> HCP1200DatasetsConfig:
        data_config = HCP1200DataConfig.from_dict(
            {
                "description": dict_["data"]["description"],
                "subjects": [
                    subject for subjects in dict_["datasets"]["dataset_subjects"].values() for subject in subjects
                ],
                "mode": dict_["data"]["mode"],
                "pes": dict_["data"]["pes"],
                "tasks": dict_["data"]["tasks"],
                "subtask_limit": dict_["data"]["subtask_limit"],
            }
        )

        config = object.__new__(HCP1200DatasetsConfig)
        DatasetsConfig_.__init__(
            config,
            description=dict_["datasets"]["description"],
            data_config=data_config,
            dataset_ids=dict_["datasets"]["dataset_subjects"],
            split=dict_["datasets"]["split"],
        )

        config.normalization = dict_["datasets"]["normalization"]
        config.window_width = dict_["datasets"]["window_width"]
        config.class_type = dict_["datasets"]["class_type"]
        config.time_augmentation = dict_["datasets"].get("time_augmentation", True)  # old version
        config.spatial_augmentation = dict_["datasets"].get("spatial_augmentation", None)  # old version

        return config

    def _as_dict(self) -> dict[str, Any]:
        return {
            "data": self.data_config.as_dict(),
            "datasets": {
                "description": self.description,
                "normalization": self.normalization,
                "window_width": self.window_width,
                "class_type": self.class_type,
                "time_augmentation": self.time_augmentation,
                "spatial_augmentation": self.spatial_augmentation,
                "split": self.split,
                "dataset_subjects": self.dataset_ids,
            },
        }

    def get_dataset_size(self, name: str) -> int:
        if name not in self.dataset_ids.keys():
            raise KeyError(f"Dataset `{name}` not found.")

        return len(self.dataset_ids[name])

    def get_classes(self, pretty: bool = False) -> list[str]:
        return self.data_config.get_classes(pretty=pretty)[self.class_type]
