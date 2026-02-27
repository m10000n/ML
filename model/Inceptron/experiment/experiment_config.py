from data import data_loader_config
from data.HCP_1200 import hcp_1200_data_config, hcp_1200_dataset_config
from model.criterion import CrossEntropyLossConfig
from model.experiment import ExperimentConfig, TrainConfig
from model.Inceptron.architecture import model_config
from model.optimizer import AdamWConfig
from model.scheduler import CosineAnnealingLRConfig
from model.stop_criterion import StopPatienceConfig
from model.warmup import WarmupConfig

# lr: 1e-3, 3e-4, 1e-4
# weight decay: 1e-3, 1e-4, 1e-5


# prep
def inceptron__prep__00() -> ExperimentConfig:  # done, CV
    return ExperimentConfig(
        description="lr: 3e-4, weight decay: 1e-3, warmup: 1000 steps, eta_min: 3e-6, batch size: 16, batch accum.: 4",
        model=model_config.inceptron(),
        criterion=CrossEntropyLossConfig(),
        train=TrainConfig(
            optimizer=AdamWConfig(lr=3e-4, betas=(0.9, 0.999), weight_decay=1e-3, exclude=["bias", "norm"]),
            scheduler=CosineAnnealingLRConfig(type="epoch", t_max=100, eta_min=3e-6),
            stop_criterion=StopPatienceConfig(patience=20),
            batch_accumulation=4,
            warmup=WarmupConfig(n_steps=1000, function="linear"),
            no_learn_limit=10,
        ),
        data=hcp_1200_dataset_config.mean_std_voxel_file_ta(hcp_1200_data_config.prep()),
        data_loaders=data_loader_config.batch_size_16(),
    )


# unp
def inceptron__unp__00() -> ExperimentConfig:  # done
    return ExperimentConfig(
        description="lr: 3e-4, weight decay: 1e-3, warmup: 1000 steps, eta_min: 3e-6, batch size: 16, batch accum.: 4",
        model=model_config.inceptron(),
        criterion=CrossEntropyLossConfig(),
        train=TrainConfig(
            optimizer=AdamWConfig(lr=3e-4, betas=(0.9, 0.999), weight_decay=1e-3, exclude=["bias", "norm"]),
            scheduler=CosineAnnealingLRConfig(type="epoch", t_max=100, eta_min=3e-6),
            stop_criterion=StopPatienceConfig(patience=20),
            batch_accumulation=4,
            warmup=WarmupConfig(n_steps=1000, function="linear"),
            no_learn_limit=10,
        ),
        data=hcp_1200_dataset_config.mean_std_file_ta(hcp_1200_data_config.unp()),
        data_loaders=data_loader_config.batch_size_16(),
    )


def inceptron__unp__01() -> ExperimentConfig:  # done, CV
    return ExperimentConfig(
        description="lr: 3e-4, weight decay: 1e-3, warmup: 1000 steps, eta_min: 3e-6, batch size: 16, batch accum.: 4",
        model=model_config.inceptron(),
        criterion=CrossEntropyLossConfig(),
        train=TrainConfig(
            optimizer=AdamWConfig(lr=3e-4, betas=(0.9, 0.999), weight_decay=1e-3, exclude=["bias", "norm"]),
            scheduler=CosineAnnealingLRConfig(type="epoch", t_max=100, eta_min=3e-6),
            stop_criterion=StopPatienceConfig(patience=20),
            batch_accumulation=4,
            warmup=WarmupConfig(n_steps=1000, function="linear"),
            no_learn_limit=10,
        ),
        data=hcp_1200_dataset_config.mean_std_file_ta_sa_00(hcp_1200_data_config.unp()),
        data_loaders=data_loader_config.batch_size_16(),
    )


def inceptron__unp__02() -> ExperimentConfig:  # done
    return ExperimentConfig(
        description="lr: 3e-4, weight decay: 1e-3, warmup: 1000 steps, eta_min: 3e-6, batch size: 16, batch accum.: 4",
        model=model_config.inceptron(),
        criterion=CrossEntropyLossConfig(),
        train=TrainConfig(
            optimizer=AdamWConfig(lr=3e-4, betas=(0.9, 0.999), weight_decay=1e-3, exclude=["bias", "norm"]),
            scheduler=CosineAnnealingLRConfig(type="epoch", t_max=100, eta_min=3e-6),
            stop_criterion=StopPatienceConfig(patience=20),
            batch_accumulation=4,
            warmup=WarmupConfig(n_steps=1000, function="linear"),
            no_learn_limit=10,
        ),
        data=hcp_1200_dataset_config.mean_std_file_ta_sa_01(hcp_1200_data_config.unp()),
        data_loaders=data_loader_config.batch_size_16(),
    )


def inceptron__unp__03() -> ExperimentConfig:  # done
    return ExperimentConfig(
        description="lr: 3e-4, weight decay: 1e-3, warmup: 1000 steps, eta_min: 3e-6, batch size: 16, batch accum.: 4",
        model=model_config.inceptron(),
        criterion=CrossEntropyLossConfig(),
        train=TrainConfig(
            optimizer=AdamWConfig(lr=3e-4, betas=(0.9, 0.999), weight_decay=1e-3, exclude=["bias", "norm"]),
            scheduler=CosineAnnealingLRConfig(type="epoch", t_max=100, eta_min=3e-6),
            stop_criterion=StopPatienceConfig(patience=20),
            batch_accumulation=4,
            warmup=WarmupConfig(n_steps=1000, function="linear"),
            no_learn_limit=10,
        ),
        data=hcp_1200_dataset_config.mean_std_file_ta_sa_02(hcp_1200_data_config.unp()),
        data_loaders=data_loader_config.batch_size_16(),
    )
