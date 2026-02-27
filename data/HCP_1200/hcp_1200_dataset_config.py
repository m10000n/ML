from typing import Literal

from data.HCP_1200.hcp_1200_dataset import HCP1200DataConfig, HCP1200DatasetsConfig
from model.experiment import EXPERIMENT_DATASETS

_WANG_SPLIT = [0.7, 0.1, 0.2]
_WANG_WINDOW_WIDTH = 27

_SPATIAL_AUGMENTATION_0: dict[Literal["angle_max_deg", "trans_max_voxel", "scale_max"], float] = {
    "angle_max_deg": 5.0,
    "trans_max_voxel": 5.0,
    "scale_max": 0.10,
}

_SPATIAL_AUGMENTATION_1: dict[Literal["angle_max_deg", "trans_max_voxel", "scale_max"], float] = {
    "angle_max_deg": 10,
    "trans_max_voxel": 10,
    "scale_max": 0.15,
}

_SPATIAL_AUGMENTATION_2: dict[Literal["angle_max_deg", "trans_max_voxel", "scale_max"], float] = {
    "angle_max_deg": 15.0,
    "trans_max_voxel": 15.0,
    "scale_max": 0.20,
}


# mean_std_window
def mean_std_window_ta(
    data_config: HCP1200DataConfig, split: list[float] = _WANG_SPLIT, dataset_names: list[str] = EXPERIMENT_DATASETS
) -> HCP1200DatasetsConfig:
    return HCP1200DatasetsConfig(
        description="normalization: mean_std_window, window width: 27, time augmentation: True",
        data_config=data_config,
        split=split,
        dataset_names=dataset_names,
        normalization="mean_std_window",
        window_width=_WANG_WINDOW_WIDTH,
        class_type="task",
        time_augmentation=True,
        spatial_augmentation=None,
    )


def mean_std_window_ta_sa_00(
    data_config: HCP1200DataConfig, split: list[float] = _WANG_SPLIT, dataset_names: list[str] = EXPERIMENT_DATASETS
) -> HCP1200DatasetsConfig:
    return HCP1200DatasetsConfig(
        description="normalization: mean_std_window; window width: 27; time augmentation: True; spatial augmentation: 5, 5, 0.1",
        data_config=data_config,
        split=split,
        dataset_names=dataset_names,
        normalization="mean_std_window",
        window_width=_WANG_WINDOW_WIDTH,
        class_type="task",
        time_augmentation=True,
        spatial_augmentation=_SPATIAL_AUGMENTATION_0,
    )


def mean_std_window_ta_sa_01(
    data_config: HCP1200DataConfig, split: list[float] = _WANG_SPLIT, dataset_names: list[str] = EXPERIMENT_DATASETS
) -> HCP1200DatasetsConfig:
    return HCP1200DatasetsConfig(
        description="normalization: mean_std_window; window width: 27; time augmentation: True; spatial augmentation: 10, 10, 0.15",
        data_config=data_config,
        split=split,
        dataset_names=dataset_names,
        normalization="mean_std_window",
        window_width=_WANG_WINDOW_WIDTH,
        class_type="task",
        time_augmentation=True,
        spatial_augmentation=_SPATIAL_AUGMENTATION_1,
    )


def mean_std_window_ta_sa_02(
    data_config: HCP1200DataConfig, split: list[float] = _WANG_SPLIT, dataset_names: list[str] = EXPERIMENT_DATASETS
) -> HCP1200DatasetsConfig:
    return HCP1200DatasetsConfig(
        description="normalization: mean_std_window; window width: 27; time augmentation: True; spatial augmentation: 15, 15, 0.2",
        data_config=data_config,
        split=split,
        dataset_names=dataset_names,
        normalization="mean_std_window",
        window_width=_WANG_WINDOW_WIDTH,
        class_type="task",
        time_augmentation=True,
        spatial_augmentation=_SPATIAL_AUGMENTATION_2,
    )


# mean_std_file
def mean_std_file_ta(
    data_config: HCP1200DataConfig, split: list[float] = _WANG_SPLIT, dataset_names: list[str] = EXPERIMENT_DATASETS
) -> HCP1200DatasetsConfig:
    return HCP1200DatasetsConfig(
        description="normalization: mean_std_file, window width: 27, time augmentation: True",
        data_config=data_config,
        split=split,
        dataset_names=dataset_names,
        normalization="mean_std_file",
        window_width=_WANG_WINDOW_WIDTH,
        class_type="task",
        time_augmentation=True,
        spatial_augmentation=None,
    )


def mean_std_file_ta_sa_00(
    data_config: HCP1200DataConfig, split: list[float] = _WANG_SPLIT, dataset_names: list[str] = EXPERIMENT_DATASETS
) -> HCP1200DatasetsConfig:
    return HCP1200DatasetsConfig(
        description="normalization: mean_std_file, window width: 27, time augmentation: True, spatial augmentation: 5, 5, 0.1",
        data_config=data_config,
        split=split,
        dataset_names=dataset_names,
        normalization="mean_std_file",
        window_width=_WANG_WINDOW_WIDTH,
        class_type="task",
        time_augmentation=True,
        spatial_augmentation=_SPATIAL_AUGMENTATION_0,
    )


def mean_std_file_ta_sa_01(
    data_config: HCP1200DataConfig, split: list[float] = _WANG_SPLIT, dataset_names: list[str] = EXPERIMENT_DATASETS
) -> HCP1200DatasetsConfig:  #
    return HCP1200DatasetsConfig(
        description="normalization: mean_std_file, window width: 27, time augmentation: True, spatial augmentation: 10, 10, 0.15",
        data_config=data_config,
        split=split,
        dataset_names=dataset_names,
        normalization="mean_std_file",
        window_width=_WANG_WINDOW_WIDTH,
        class_type="task",
        time_augmentation=True,
        spatial_augmentation=_SPATIAL_AUGMENTATION_1,
    )


def mean_std_file_ta_sa_02(
    data_config: HCP1200DataConfig, split: list[float] = _WANG_SPLIT, dataset_names: list[str] = EXPERIMENT_DATASETS
) -> HCP1200DatasetsConfig:
    return HCP1200DatasetsConfig(
        description="normalization: mean_std_file, window width: 27, time augmentation: True, spatial augmentation: 15, 15, 0.2",
        data_config=data_config,
        split=split,
        dataset_names=dataset_names,
        normalization="mean_std_file",
        window_width=_WANG_WINDOW_WIDTH,
        class_type="task",
        time_augmentation=True,
        spatial_augmentation=_SPATIAL_AUGMENTATION_2,
    )


# mean_std_voxel_window
def mean_std_voxel_window_ta(
    data_config: HCP1200DataConfig, split: list[float] = _WANG_SPLIT, dataset_names: list[str] = EXPERIMENT_DATASETS
) -> HCP1200DatasetsConfig:
    return HCP1200DatasetsConfig(
        description="normalization: mean_std_voxel_window, window width: 27, time augmentation: True",
        data_config=data_config,
        split=split,
        dataset_names=dataset_names,
        normalization="mean_std_voxel_window",
        window_width=_WANG_WINDOW_WIDTH,
        class_type="task",
        time_augmentation=True,
        spatial_augmentation=None,
    )


# mean_std_voxel_file
def mean_std_voxel_file_ta(
    data_config: HCP1200DataConfig, split: list[float] = _WANG_SPLIT, dataset_names: list[str] = EXPERIMENT_DATASETS
) -> HCP1200DatasetsConfig:
    return HCP1200DatasetsConfig(
        description="normalization: mean_std_voxel_file, window width: 27, time augmentation: True",
        data_config=data_config,
        split=split,
        dataset_names=dataset_names,
        normalization="mean_std_voxel_file",
        window_width=_WANG_WINDOW_WIDTH,
        class_type="task",
        time_augmentation=True,
        spatial_augmentation=None,
    )


# min_max_window
def min_max_window_ta(
    data_config: HCP1200DataConfig, split: list[float] = _WANG_SPLIT, dataset_names: list[str] = EXPERIMENT_DATASETS
) -> HCP1200DatasetsConfig:
    return HCP1200DatasetsConfig(
        description="normalization: min_max_window, window width: 27, time augmentation: True",
        data_config=data_config,
        split=split,
        dataset_names=dataset_names,
        normalization="min_max_window",
        window_width=_WANG_WINDOW_WIDTH,
        class_type="task",
        time_augmentation=True,
        spatial_augmentation=None,
    )


def min_max_window_ta_sa_00(
    data_config: HCP1200DataConfig, split: list[float] = _WANG_SPLIT, dataset_names: list[str] = EXPERIMENT_DATASETS
) -> HCP1200DatasetsConfig:
    return HCP1200DatasetsConfig(
        description="normalization: min_max_window, window width: 27, time augmentation: True, spatial augmentation: 5, 5, 0.1",
        data_config=data_config,
        split=split,
        dataset_names=dataset_names,
        normalization="min_max_window",
        window_width=_WANG_WINDOW_WIDTH,
        class_type="task",
        time_augmentation=True,
        spatial_augmentation=_SPATIAL_AUGMENTATION_0,
    )


def min_max_window_ta_sa_01(
    data_config: HCP1200DataConfig, split: list[float] = _WANG_SPLIT, dataset_names: list[str] = EXPERIMENT_DATASETS
) -> HCP1200DatasetsConfig:
    return HCP1200DatasetsConfig(
        description="normalization: min_max_window, window width: 27, time augmentation: True, spatial augmentation: 10, 10, 0.15",
        data_config=data_config,
        split=split,
        dataset_names=dataset_names,
        normalization="min_max_window",
        window_width=_WANG_WINDOW_WIDTH,
        class_type="task",
        time_augmentation=True,
        spatial_augmentation=_SPATIAL_AUGMENTATION_1,
    )


def min_max_window_ta_sa_02(
    data_config: HCP1200DataConfig, split: list[float] = _WANG_SPLIT, dataset_names: list[str] = EXPERIMENT_DATASETS
) -> HCP1200DatasetsConfig:
    return HCP1200DatasetsConfig(
        description="normalization: min_max_window, window width: 27, time augmentation: True, spatial augmentation: 15, 15, 0.2",
        data_config=data_config,
        split=split,
        dataset_names=dataset_names,
        normalization="min_max_window",
        window_width=_WANG_WINDOW_WIDTH,
        class_type="task",
        time_augmentation=True,
        spatial_augmentation=_SPATIAL_AUGMENTATION_2,
    )


# min_max_file
def min_max_file_ta(
    data_config: HCP1200DataConfig, split: list[float] = _WANG_SPLIT, dataset_names: list[str] = EXPERIMENT_DATASETS
) -> HCP1200DatasetsConfig:
    return HCP1200DatasetsConfig(
        description="normalization: min_max_file, window width: 27, time augmentation: True",
        data_config=data_config,
        split=split,
        dataset_names=dataset_names,
        normalization="min_max_file",
        window_width=_WANG_WINDOW_WIDTH,
        class_type="task",
        time_augmentation=True,
        spatial_augmentation=None,
    )


def min_max_file_ta_sa_00(
    data_config: HCP1200DataConfig, split: list[float] = _WANG_SPLIT, dataset_names: list[str] = EXPERIMENT_DATASETS
) -> HCP1200DatasetsConfig:
    return HCP1200DatasetsConfig(
        description="normalization: min_max_file, window width: 27, time augmentation: True, spatial augmentation: 5, 5, 0.1",
        data_config=data_config,
        split=split,
        dataset_names=dataset_names,
        normalization="min_max_file",
        window_width=_WANG_WINDOW_WIDTH,
        class_type="task",
        time_augmentation=True,
        spatial_augmentation=_SPATIAL_AUGMENTATION_0,
    )


def min_max_file_ta_sa_01(
    data_config: HCP1200DataConfig, split: list[float] = _WANG_SPLIT, dataset_names: list[str] = EXPERIMENT_DATASETS
) -> HCP1200DatasetsConfig:
    return HCP1200DatasetsConfig(
        description="normalization: min_max_file, window width: 27, time augmentation: True, spatial augmentation: 10, 10, 0.15",
        data_config=data_config,
        split=split,
        dataset_names=dataset_names,
        normalization="min_max_file",
        window_width=_WANG_WINDOW_WIDTH,
        class_type="task",
        time_augmentation=True,
        spatial_augmentation=_SPATIAL_AUGMENTATION_1,
    )


def min_max_file_ta_sa_02(
    data_config: HCP1200DataConfig, split: list[float] = _WANG_SPLIT, dataset_names: list[str] = EXPERIMENT_DATASETS
) -> HCP1200DatasetsConfig:
    return HCP1200DatasetsConfig(
        description="normalization: min_max_file, window width: 27, time augmentation: True, spatial augmentation: 15, 15, 0.2",
        data_config=data_config,
        split=split,
        dataset_names=dataset_names,
        normalization="min_max_file",
        window_width=_WANG_WINDOW_WIDTH,
        class_type="task",
        time_augmentation=True,
        spatial_augmentation=_SPATIAL_AUGMENTATION_2,
    )


# min_max_voxel_window
def min_max_voxel_window_ta(
    data_config: HCP1200DataConfig, split: list[float] = _WANG_SPLIT, dataset_names: list[str] = EXPERIMENT_DATASETS
) -> HCP1200DatasetsConfig:
    return HCP1200DatasetsConfig(
        description="normalization: min_max_voxel_window, window width: 27, time augmentation: True",
        data_config=data_config,
        split=split,
        dataset_names=dataset_names,
        normalization="min_max_voxel_window",
        window_width=_WANG_WINDOW_WIDTH,
        class_type="task",
        time_augmentation=True,
        spatial_augmentation=None,
    )


# min_max_voxel_file
def min_max_voxel_file_ta(
    data_config: HCP1200DataConfig, split: list[float] = _WANG_SPLIT, dataset_names: list[str] = EXPERIMENT_DATASETS
) -> HCP1200DatasetsConfig:
    return HCP1200DatasetsConfig(
        description="normalization: min_max_voxel_file, window width: 27, time augmentation: True",
        data_config=data_config,
        split=split,
        dataset_names=dataset_names,
        normalization="min_max_voxel_file",
        window_width=_WANG_WINDOW_WIDTH,
        class_type="task",
        time_augmentation=True,
        spatial_augmentation=None,
    )
