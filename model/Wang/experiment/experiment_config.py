from data import data_loader_config
from data.HCP_1200 import hcp_1200_data_config, hcp_1200_dataset_config
from model.criterion import CrossEntropyLossConfig
from model.experiment import ExperimentConfig, TrainConfig
from model.optimizer import AdamConfig
from model.scheduler import ReduceLROnPlateauConfig
from model.stop_criterion import StopPatienceConfig
from model.Wang.architecture import model_config as model_config

_CRITERION_ORIGINAL = CrossEntropyLossConfig()
_OPTIMIZER_ORIGINAL = AdamConfig(lr=0.001, betas=(0.9, 0.999))
_SCHEDULER_ORIGINAL = ReduceLROnPlateauConfig(mode="min", factor=0.1, patience=15)
_STOP_CRITERION_ORIGINAL = StopPatienceConfig(patience=30)
_DATA_LOADER_ORIGINAL = data_loader_config.batch_size_32

_DESCRIPTION_ORIGINAL = "original hyperparameters"
_NO_LEARN_LIMIT = 10
_TRAIN_CONFIG_ORIGINAL = TrainConfig(
    optimizer=_OPTIMIZER_ORIGINAL,
    scheduler=_SCHEDULER_ORIGINAL,
    stop_criterion=_STOP_CRITERION_ORIGINAL,
    no_learn_limit=_NO_LEARN_LIMIT,
)
_DATA_PREP = hcp_1200_data_config.prep()


# 00 -> mean_std_window
# 01 -> mean_std_file
# 02 -> mean_std_voxel_window
# 03 -> mean_std_voxel_file
# 04 -> min_max_window
# 05 -> min_max_file
# 06 -> min_max_voxel_window
# 07 -> min_max_voxel_file


# original
## prep
def original_prep__00() -> ExperimentConfig:  # done x2
    return ExperimentConfig(
        description=_DESCRIPTION_ORIGINAL,
        model=model_config.original_prep(),
        criterion=_CRITERION_ORIGINAL,
        train=_TRAIN_CONFIG_ORIGINAL,
        data=hcp_1200_dataset_config.mean_std_window_ta(_DATA_PREP),
        data_loaders=_DATA_LOADER_ORIGINAL(),
    )


def original_prep__01() -> ExperimentConfig:  # done
    return ExperimentConfig(
        description=_DESCRIPTION_ORIGINAL,
        model=model_config.original_prep(),
        criterion=_CRITERION_ORIGINAL,
        train=_TRAIN_CONFIG_ORIGINAL,
        data=hcp_1200_dataset_config.mean_std_file_ta(_DATA_PREP),
        data_loaders=_DATA_LOADER_ORIGINAL(),
    )


def original_prep__02() -> ExperimentConfig:  # done
    return ExperimentConfig(
        description=_DESCRIPTION_ORIGINAL,
        model=model_config.original_prep(),
        criterion=_CRITERION_ORIGINAL,
        train=_TRAIN_CONFIG_ORIGINAL,
        data=hcp_1200_dataset_config.mean_std_voxel_window_ta(_DATA_PREP),
        data_loaders=_DATA_LOADER_ORIGINAL(),
    )


def original_prep__03() -> ExperimentConfig:  # done
    return ExperimentConfig(
        description=_DESCRIPTION_ORIGINAL,
        model=model_config.original_prep(),
        criterion=_CRITERION_ORIGINAL,
        train=_TRAIN_CONFIG_ORIGINAL,
        data=hcp_1200_dataset_config.mean_std_voxel_file_ta(_DATA_PREP),
        data_loaders=_DATA_LOADER_ORIGINAL(),
    )


def original_prep__04() -> ExperimentConfig:  # done
    return ExperimentConfig(
        description=_DESCRIPTION_ORIGINAL,
        model=model_config.original_prep(),
        criterion=_CRITERION_ORIGINAL,
        train=_TRAIN_CONFIG_ORIGINAL,
        data=hcp_1200_dataset_config.min_max_window_ta(_DATA_PREP),
        data_loaders=_DATA_LOADER_ORIGINAL(),
    )


def original_prep__05() -> ExperimentConfig:  # done
    return ExperimentConfig(
        description=_DESCRIPTION_ORIGINAL,
        model=model_config.original_prep(),
        criterion=_CRITERION_ORIGINAL,
        train=_TRAIN_CONFIG_ORIGINAL,
        data=hcp_1200_dataset_config.min_max_file_ta(_DATA_PREP),
        data_loaders=_DATA_LOADER_ORIGINAL(),
    )


def original_prep__06() -> ExperimentConfig:  # done
    return ExperimentConfig(
        description=_DESCRIPTION_ORIGINAL,
        model=model_config.original_prep(),
        criterion=_CRITERION_ORIGINAL,
        train=_TRAIN_CONFIG_ORIGINAL,
        data=hcp_1200_dataset_config.min_max_voxel_window_ta(_DATA_PREP),
        data_loaders=_DATA_LOADER_ORIGINAL(),
    )


def original_prep__07() -> ExperimentConfig:  # done
    return ExperimentConfig(
        description=_DESCRIPTION_ORIGINAL,
        model=model_config.original_prep(),
        criterion=_CRITERION_ORIGINAL,
        train=_TRAIN_CONFIG_ORIGINAL,
        data=hcp_1200_dataset_config.min_max_voxel_file_ta(_DATA_PREP),
        data_loaders=_DATA_LOADER_ORIGINAL(),
    )


# original2
## prep
def original2_prep__00() -> ExperimentConfig:  # done
    return ExperimentConfig(
        description=_DESCRIPTION_ORIGINAL,
        model=model_config.original2_prep(),
        criterion=_CRITERION_ORIGINAL,
        train=_TRAIN_CONFIG_ORIGINAL,
        data=hcp_1200_dataset_config.mean_std_window_ta(_DATA_PREP),
        data_loaders=_DATA_LOADER_ORIGINAL(),
    )


def original2_prep__01() -> ExperimentConfig:  # done x2
    return ExperimentConfig(
        description=_DESCRIPTION_ORIGINAL,
        model=model_config.original2_prep(),
        criterion=_CRITERION_ORIGINAL,
        train=_TRAIN_CONFIG_ORIGINAL,
        data=hcp_1200_dataset_config.mean_std_file_ta(_DATA_PREP),
        data_loaders=_DATA_LOADER_ORIGINAL(),
    )


def original2_prep__02() -> ExperimentConfig:  # done
    return ExperimentConfig(
        description=_DESCRIPTION_ORIGINAL,
        model=model_config.original2_prep(),
        criterion=_CRITERION_ORIGINAL,
        train=_TRAIN_CONFIG_ORIGINAL,
        data=hcp_1200_dataset_config.mean_std_voxel_window_ta(_DATA_PREP),
        data_loaders=_DATA_LOADER_ORIGINAL(),
    )


def original2_prep__03() -> ExperimentConfig:  # done
    return ExperimentConfig(
        description=_DESCRIPTION_ORIGINAL,
        model=model_config.original2_prep(),
        criterion=_CRITERION_ORIGINAL,
        train=_TRAIN_CONFIG_ORIGINAL,
        data=hcp_1200_dataset_config.mean_std_voxel_file_ta(_DATA_PREP),
        data_loaders=_DATA_LOADER_ORIGINAL(),
    )


def original2_prep__04() -> ExperimentConfig:  # done x2
    return ExperimentConfig(
        description=_DESCRIPTION_ORIGINAL,
        model=model_config.original2_prep(),
        criterion=_CRITERION_ORIGINAL,
        train=_TRAIN_CONFIG_ORIGINAL,
        data=hcp_1200_dataset_config.min_max_window_ta(_DATA_PREP),
        data_loaders=_DATA_LOADER_ORIGINAL(),
    )


def original2_prep__05() -> ExperimentConfig:  # done
    return ExperimentConfig(
        description=_DESCRIPTION_ORIGINAL,
        model=model_config.original2_prep(),
        criterion=_CRITERION_ORIGINAL,
        train=_TRAIN_CONFIG_ORIGINAL,
        data=hcp_1200_dataset_config.min_max_file_ta(_DATA_PREP),
        data_loaders=_DATA_LOADER_ORIGINAL(),
    )


def original2_prep__06() -> ExperimentConfig:  # done
    return ExperimentConfig(
        description=_DESCRIPTION_ORIGINAL,
        model=model_config.original2_prep(),
        criterion=_CRITERION_ORIGINAL,
        train=_TRAIN_CONFIG_ORIGINAL,
        data=hcp_1200_dataset_config.min_max_voxel_window_ta(_DATA_PREP),
        data_loaders=_DATA_LOADER_ORIGINAL(),
    )


def original2_prep__07() -> ExperimentConfig:  # done
    return ExperimentConfig(
        description=_DESCRIPTION_ORIGINAL,
        model=model_config.original2_prep(),
        criterion=_CRITERION_ORIGINAL,
        train=_TRAIN_CONFIG_ORIGINAL,
        data=hcp_1200_dataset_config.min_max_voxel_file_ta(_DATA_PREP),
        data_loaders=_DATA_LOADER_ORIGINAL(),
    )
