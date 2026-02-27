from data import data_loader_config
from data.HCP_1200 import hcp_1200_data_config, hcp_1200_dataset_config
from model.BrT.architecture import model_config
from model.criterion import CrossEntropyLossConfig
from model.experiment import ExperimentConfig, TrainConfig
from model.optimizer import AdamWConfig
from model.scheduler import CosineAnnealingLRConfig
from model.stop_criterion import StopPatienceConfig
from model.warmup import WarmupConfig


# brt_small
## prep
### t4
def brt_small_t4p_prep__00() -> ExperimentConfig:  # done
    return ExperimentConfig(
        description="lr: 1e-4, weight decay: 1e-2, warmup: 1000 steps, eta_min: 1e-6, batch size: 16, batch accum.: 4",
        model=model_config.brt_small_t4p_prep(),
        criterion=CrossEntropyLossConfig(),
        train=TrainConfig(
            optimizer=AdamWConfig(
                lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-2, exclude=["bias", "norm", "class_token", "pos_embed"]
            ),
            scheduler=CosineAnnealingLRConfig(type="epoch", t_max=100, eta_min=1e-6),
            stop_criterion=StopPatienceConfig(patience=20),
            batch_accumulation=4,
            warmup=WarmupConfig(n_steps=1000, function="linear"),
            no_learn_limit=10,
        ),
        data=hcp_1200_dataset_config.mean_std_voxel_file_ta(hcp_1200_data_config.prep()),
        data_loaders=data_loader_config.batch_size_16(),
    )


def brt_small_t4c1_prep__00() -> ExperimentConfig:  # done
    return ExperimentConfig(
        description="lr: 1e-4, weight decay: 1e-2, warmup: 1000 steps, eta_min: 1e-6, batch size: 16, batch accum.: 4",
        model=model_config.brt_small_t4c1_prep(),
        criterion=CrossEntropyLossConfig(),
        train=TrainConfig(
            optimizer=AdamWConfig(
                lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-2, exclude=["bias", "norm", "class_token", "pos_embed"]
            ),
            scheduler=CosineAnnealingLRConfig(type="epoch", t_max=100, eta_min=1e-6),
            stop_criterion=StopPatienceConfig(patience=20),
            batch_accumulation=4,
            warmup=WarmupConfig(n_steps=1000, function="linear"),
            no_learn_limit=10,
        ),
        data=hcp_1200_dataset_config.mean_std_voxel_file_ta(hcp_1200_data_config.prep()),
        data_loaders=data_loader_config.batch_size_16(),
    )


def brt_small_t4c2_prep__00() -> ExperimentConfig:  # done
    return ExperimentConfig(
        description="lr: 1e-4, weight decay: 1e-2, warmup: 1000 steps, eta_min: 1e-6, batch size: 16, batch accum.: 4",
        model=model_config.brt_small_t4c2_prep(),
        criterion=CrossEntropyLossConfig(),
        train=TrainConfig(
            optimizer=AdamWConfig(
                lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-2, exclude=["bias", "norm", "class_token", "pos_embed"]
            ),
            scheduler=CosineAnnealingLRConfig(type="epoch", t_max=100, eta_min=1e-6),
            stop_criterion=StopPatienceConfig(patience=20),
            batch_accumulation=4,
            warmup=WarmupConfig(n_steps=1000, function="linear"),
            no_learn_limit=10,
        ),
        data=hcp_1200_dataset_config.mean_std_voxel_file_ta(hcp_1200_data_config.prep()),
        data_loaders=data_loader_config.batch_size_16(),
    )


### t8
def brt_small_t8p_prep__00() -> ExperimentConfig:  # done
    return ExperimentConfig(
        description="lr: 1e-4, weight decay: 1e-2, warmup: 1000 steps, eta_min: 1e-6, batch size: 16, batch accum.: 4",
        model=model_config.brt_small_t8p_prep(),
        criterion=CrossEntropyLossConfig(),
        train=TrainConfig(
            optimizer=AdamWConfig(
                lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-2, exclude=["bias", "norm", "class_token", "pos_embed"]
            ),
            scheduler=CosineAnnealingLRConfig(type="epoch", t_max=100, eta_min=1e-6),
            stop_criterion=StopPatienceConfig(patience=20),
            batch_accumulation=4,
            warmup=WarmupConfig(n_steps=1000, function="linear"),
            no_learn_limit=10,
        ),
        data=hcp_1200_dataset_config.mean_std_voxel_file_ta(hcp_1200_data_config.prep()),
        data_loaders=data_loader_config.batch_size_16(),
    )


def brt_small_t8c1_prep__00() -> ExperimentConfig:  # done
    return ExperimentConfig(
        description="lr: 1e-4, weight decay: 1e-2, warmup: 1000 steps, eta_min: 1e-6, batch size: 16, batch accum.: 4",
        model=model_config.brt_small_t8c1_prep(),
        criterion=CrossEntropyLossConfig(),
        train=TrainConfig(
            optimizer=AdamWConfig(
                lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-2, exclude=["bias", "norm", "class_token", "pos_embed"]
            ),
            scheduler=CosineAnnealingLRConfig(type="epoch", t_max=100, eta_min=1e-6),
            stop_criterion=StopPatienceConfig(patience=20),
            batch_accumulation=4,
            warmup=WarmupConfig(n_steps=1000, function="linear"),
            no_learn_limit=10,
        ),
        data=hcp_1200_dataset_config.mean_std_voxel_file_ta(hcp_1200_data_config.prep()),
        data_loaders=data_loader_config.batch_size_16(),
    )


def brt_small_t8c2_prep__00() -> ExperimentConfig:  # done
    return ExperimentConfig(
        description="lr: 1e-4, weight decay: 1e-2, warmup: 1000 steps, eta_min: 1e-6, batch size: 16, batch accum.: 4",
        model=model_config.brt_small_t8c2_prep(),
        criterion=CrossEntropyLossConfig(),
        train=TrainConfig(
            optimizer=AdamWConfig(
                lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-2, exclude=["bias", "norm", "class_token", "pos_embed"]
            ),
            scheduler=CosineAnnealingLRConfig(type="epoch", t_max=100, eta_min=1e-6),
            stop_criterion=StopPatienceConfig(patience=20),
            batch_accumulation=4,
            warmup=WarmupConfig(n_steps=1000, function="linear"),
            no_learn_limit=10,
        ),
        data=hcp_1200_dataset_config.mean_std_voxel_file_ta(hcp_1200_data_config.prep()),
        data_loaders=data_loader_config.batch_size_16(),
    )


### v/4
def brt_small_vp4_prep__00() -> ExperimentConfig:  # done
    return ExperimentConfig(
        description="lr: 1e-4, weight decay: 1e-2, warmup: 1000 steps, eta_min: 1e-6, batch size: 16, batch accum.: 4",
        model=model_config.brt_small_vp4_prep(),
        criterion=CrossEntropyLossConfig(),
        train=TrainConfig(
            optimizer=AdamWConfig(
                lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-2, exclude=["bias", "norm", "class_token", "pos_embed"]
            ),
            scheduler=CosineAnnealingLRConfig(type="epoch", t_max=100, eta_min=1e-6),
            stop_criterion=StopPatienceConfig(patience=20),
            batch_accumulation=4,
            warmup=WarmupConfig(n_steps=1000, function="linear"),
            no_learn_limit=10,
        ),
        data=hcp_1200_dataset_config.mean_std_voxel_file_ta(hcp_1200_data_config.prep()),
        data_loaders=data_loader_config.batch_size_16(),
    )


def brt_small_vc4_prep__00() -> ExperimentConfig:  # done
    return ExperimentConfig(
        description="lr: 1e-4, weight decay: 1e-2, warmup: 1000 steps, eta_min: 1e-6, batch size: 16, batch accum.: 4",
        model=model_config.brt_small_vc4_prep(),
        criterion=CrossEntropyLossConfig(),
        train=TrainConfig(
            optimizer=AdamWConfig(
                lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-2, exclude=["bias", "norm", "class_token", "pos_embed"]
            ),
            scheduler=CosineAnnealingLRConfig(type="epoch", t_max=100, eta_min=1e-6),
            stop_criterion=StopPatienceConfig(patience=20),
            batch_accumulation=4,
            warmup=WarmupConfig(n_steps=1000, function="linear"),
            no_learn_limit=10,
        ),
        data=hcp_1200_dataset_config.mean_std_voxel_file_ta(hcp_1200_data_config.prep()),
        data_loaders=data_loader_config.batch_size_16(),
    )


## unp
### t4
def brt_small_t4p_unp__00() -> ExperimentConfig:  # done
    return ExperimentConfig(
        description="lr: 1e-4, weight decay: 1e-2, warmup: 1000 steps, eta_min: 1e-6, batch size: 16, batch accum.: 4",
        model=model_config.brt_small_t4p_unp(),
        criterion=CrossEntropyLossConfig(),
        train=TrainConfig(
            optimizer=AdamWConfig(
                lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-2, exclude=["bias", "norm", "class_token", "pos_embed"]
            ),
            scheduler=CosineAnnealingLRConfig(type="epoch", t_max=100, eta_min=1e-6),
            stop_criterion=StopPatienceConfig(patience=20),
            batch_accumulation=4,
            warmup=WarmupConfig(n_steps=1000, function="linear"),
            no_learn_limit=10,
        ),
        data=hcp_1200_dataset_config.mean_std_file_ta(hcp_1200_data_config.unp()),
        data_loaders=data_loader_config.batch_size_16(),
    )


def brt_small_t4p_unp__01() -> ExperimentConfig:  # done
    return ExperimentConfig(
        description="lr: 1e-4, weight decay: 1e-2, warmup: 1000 steps, eta_min: 1e-6, batch size: 16, batch accum.: 4",
        model=model_config.brt_small_t4p_unp(),
        criterion=CrossEntropyLossConfig(),
        train=TrainConfig(
            optimizer=AdamWConfig(
                lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-2, exclude=["bias", "norm", "class_token", "pos_embed"]
            ),
            scheduler=CosineAnnealingLRConfig(type="epoch", t_max=100, eta_min=1e-6),
            stop_criterion=StopPatienceConfig(patience=20),
            batch_accumulation=4,
            warmup=WarmupConfig(n_steps=1000, function="linear"),
            no_learn_limit=10,
        ),
        data=hcp_1200_dataset_config.mean_std_file_ta_sa_00(hcp_1200_data_config.unp()),
        data_loaders=data_loader_config.batch_size_16(),
    )


def brt_small_t4c1_unp__00() -> ExperimentConfig:  # done
    return ExperimentConfig(
        description="lr: 1e-4, weight decay: 1e-2, warmup: 1000 steps, eta_min: 1e-6, batch size: 16, batch accum.: 4",
        model=model_config.brt_small_t4c1_unp(),
        criterion=CrossEntropyLossConfig(),
        train=TrainConfig(
            optimizer=AdamWConfig(
                lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-2, exclude=["bias", "norm", "class_token", "pos_embed"]
            ),
            scheduler=CosineAnnealingLRConfig(type="epoch", t_max=100, eta_min=1e-6),
            stop_criterion=StopPatienceConfig(patience=20),
            batch_accumulation=4,
            warmup=WarmupConfig(n_steps=1000, function="linear"),
            no_learn_limit=10,
        ),
        data=hcp_1200_dataset_config.mean_std_file_ta(hcp_1200_data_config.unp()),
        data_loaders=data_loader_config.batch_size_16(),
    )


def brt_small_t4c1_unp__01() -> ExperimentConfig:  # done
    return ExperimentConfig(
        description="lr: 1e-4, weight decay: 1e-2, warmup: 1000 steps, eta_min: 1e-6, batch size: 16, batch accum.: 4",
        model=model_config.brt_small_t4c1_unp(),
        criterion=CrossEntropyLossConfig(),
        train=TrainConfig(
            optimizer=AdamWConfig(
                lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-2, exclude=["bias", "norm", "class_token", "pos_embed"]
            ),
            scheduler=CosineAnnealingLRConfig(type="epoch", t_max=100, eta_min=1e-6),
            stop_criterion=StopPatienceConfig(patience=20),
            batch_accumulation=4,
            warmup=WarmupConfig(n_steps=1000, function="linear"),
            no_learn_limit=10,
        ),
        data=hcp_1200_dataset_config.mean_std_file_ta_sa_00(hcp_1200_data_config.unp()),
        data_loaders=data_loader_config.batch_size_16(),
    )


def brt_small_t4c2_unp__00() -> ExperimentConfig:  # done
    return ExperimentConfig(
        description="lr: 1e-4, weight decay: 1e-2, warmup: 1000 steps, eta_min: 1e-6, batch size: 16, batch accum.: 4",
        model=model_config.brt_small_t4c2_unp(),
        criterion=CrossEntropyLossConfig(),
        train=TrainConfig(
            optimizer=AdamWConfig(
                lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-2, exclude=["bias", "norm", "class_token", "pos_embed"]
            ),
            scheduler=CosineAnnealingLRConfig(type="epoch", t_max=100, eta_min=1e-6),
            stop_criterion=StopPatienceConfig(patience=20),
            batch_accumulation=4,
            warmup=WarmupConfig(n_steps=1000, function="linear"),
            no_learn_limit=10,
        ),
        data=hcp_1200_dataset_config.mean_std_file_ta(hcp_1200_data_config.unp()),
        data_loaders=data_loader_config.batch_size_16(),
    )


def brt_small_t4c2_unp__01() -> ExperimentConfig:  # done, CV
    return ExperimentConfig(
        description="lr: 1e-4, weight decay: 1e-2, warmup: 1000 steps, eta_min: 1e-6, batch size: 16, batch accum.: 4",
        model=model_config.brt_small_t4c2_unp(),
        criterion=CrossEntropyLossConfig(),
        train=TrainConfig(
            optimizer=AdamWConfig(
                lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-2, exclude=["bias", "norm", "class_token", "pos_embed"]
            ),
            scheduler=CosineAnnealingLRConfig(type="epoch", t_max=100, eta_min=1e-6),
            stop_criterion=StopPatienceConfig(patience=20),
            batch_accumulation=4,
            warmup=WarmupConfig(n_steps=1000, function="linear"),
            no_learn_limit=10,
        ),
        data=hcp_1200_dataset_config.mean_std_file_ta_sa_00(hcp_1200_data_config.unp()),
        data_loaders=data_loader_config.batch_size_16(),
    )


def brt_small_t4c2_unp__02() -> ExperimentConfig:  # done
    return ExperimentConfig(
        description="lr: 1e-4, weight decay: 1e-2, warmup: 1000 steps, eta_min: 1e-6, batch size: 16, batch accum.: 4",
        model=model_config.brt_small_t4c2_unp(),
        criterion=CrossEntropyLossConfig(),
        train=TrainConfig(
            optimizer=AdamWConfig(
                lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-2, exclude=["bias", "norm", "class_token", "pos_embed"]
            ),
            scheduler=CosineAnnealingLRConfig(type="epoch", t_max=100, eta_min=1e-6),
            stop_criterion=StopPatienceConfig(patience=20),
            batch_accumulation=4,
            warmup=WarmupConfig(n_steps=1000, function="linear"),
            no_learn_limit=10,
        ),
        data=hcp_1200_dataset_config.mean_std_file_ta_sa_01(hcp_1200_data_config.unp()),
        data_loaders=data_loader_config.batch_size_16(),
    )


### t8
def brt_small_t8p_unp__00() -> ExperimentConfig:  # done
    return ExperimentConfig(
        description="lr: 1e-4, weight decay: 1e-2, warmup: 1000 steps, eta_min: 1e-6, batch size: 16, batch accum.: 4",
        model=model_config.brt_small_t8p_unp(),
        criterion=CrossEntropyLossConfig(),
        train=TrainConfig(
            optimizer=AdamWConfig(
                lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-2, exclude=["bias", "norm", "class_token", "pos_embed"]
            ),
            scheduler=CosineAnnealingLRConfig(type="epoch", t_max=100, eta_min=1e-6),
            stop_criterion=StopPatienceConfig(patience=20),
            batch_accumulation=4,
            warmup=WarmupConfig(n_steps=1000, function="linear"),
            no_learn_limit=10,
        ),
        data=hcp_1200_dataset_config.mean_std_file_ta(hcp_1200_data_config.unp()),
        data_loaders=data_loader_config.batch_size_16(),
    )


def brt_small_t8p_unp__01() -> ExperimentConfig:  # done
    return ExperimentConfig(
        description="lr: 1e-4, weight decay: 1e-2, warmup: 1000 steps, eta_min: 1e-6, batch size: 16, batch accum.: 4",
        model=model_config.brt_small_t8p_unp(),
        criterion=CrossEntropyLossConfig(),
        train=TrainConfig(
            optimizer=AdamWConfig(
                lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-2, exclude=["bias", "norm", "class_token", "pos_embed"]
            ),
            scheduler=CosineAnnealingLRConfig(type="epoch", t_max=100, eta_min=1e-6),
            stop_criterion=StopPatienceConfig(patience=20),
            batch_accumulation=4,
            warmup=WarmupConfig(n_steps=1000, function="linear"),
            no_learn_limit=10,
        ),
        data=hcp_1200_dataset_config.mean_std_file_ta_sa_00(hcp_1200_data_config.unp()),
        data_loaders=data_loader_config.batch_size_16(),
    )


def brt_small_t8c1_unp__00() -> ExperimentConfig:  # done
    return ExperimentConfig(
        description="lr: 1e-4, weight decay: 1e-2, warmup: 1000 steps, eta_min: 1e-6, batch size: 16, batch accum.: 4",
        model=model_config.brt_small_t8c1_unp(),
        criterion=CrossEntropyLossConfig(),
        train=TrainConfig(
            optimizer=AdamWConfig(
                lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-2, exclude=["bias", "norm", "class_token", "pos_embed"]
            ),
            scheduler=CosineAnnealingLRConfig(type="epoch", t_max=100, eta_min=1e-6),
            stop_criterion=StopPatienceConfig(patience=20),
            batch_accumulation=4,
            warmup=WarmupConfig(n_steps=1000, function="linear"),
            no_learn_limit=10,
        ),
        data=hcp_1200_dataset_config.mean_std_file_ta(hcp_1200_data_config.unp()),
        data_loaders=data_loader_config.batch_size_16(),
    )


def brt_small_t8c1_unp__01() -> ExperimentConfig:  # done
    return ExperimentConfig(
        description="lr: 1e-4, weight decay: 1e-2, warmup: 1000 steps, eta_min: 1e-6, batch size: 16, batch accum.: 4",
        model=model_config.brt_small_t8c1_unp(),
        criterion=CrossEntropyLossConfig(),
        train=TrainConfig(
            optimizer=AdamWConfig(
                lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-2, exclude=["bias", "norm", "class_token", "pos_embed"]
            ),
            scheduler=CosineAnnealingLRConfig(type="epoch", t_max=100, eta_min=1e-6),
            stop_criterion=StopPatienceConfig(patience=20),
            batch_accumulation=4,
            warmup=WarmupConfig(n_steps=1000, function="linear"),
            no_learn_limit=10,
        ),
        data=hcp_1200_dataset_config.mean_std_file_ta_sa_00(hcp_1200_data_config.unp()),
        data_loaders=data_loader_config.batch_size_16(),
    )


def brt_small_t8c2_unp__00() -> ExperimentConfig:  # done
    return ExperimentConfig(
        description="lr: 1e-4, weight decay: 1e-2, warmup: 1000 steps, eta_min: 1e-6, batch size: 16, batch accum.: 4",
        model=model_config.brt_small_t8c2_unp(),
        criterion=CrossEntropyLossConfig(),
        train=TrainConfig(
            optimizer=AdamWConfig(
                lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-2, exclude=["bias", "norm", "class_token", "pos_embed"]
            ),
            scheduler=CosineAnnealingLRConfig(type="epoch", t_max=100, eta_min=1e-6),
            stop_criterion=StopPatienceConfig(patience=20),
            batch_accumulation=4,
            warmup=WarmupConfig(n_steps=1000, function="linear"),
            no_learn_limit=10,
        ),
        data=hcp_1200_dataset_config.mean_std_file_ta(hcp_1200_data_config.unp()),
        data_loaders=data_loader_config.batch_size_16(),
    )


def brt_small_t8c2_unp__01() -> ExperimentConfig:  # done
    return ExperimentConfig(
        description="lr: 1e-4, weight decay: 1e-2, warmup: 1000 steps, eta_min: 1e-6, batch size: 16, batch accum.: 4",
        model=model_config.brt_small_t8c2_unp(),
        criterion=CrossEntropyLossConfig(),
        train=TrainConfig(
            optimizer=AdamWConfig(
                lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-2, exclude=["bias", "norm", "class_token", "pos_embed"]
            ),
            scheduler=CosineAnnealingLRConfig(type="epoch", t_max=100, eta_min=1e-6),
            stop_criterion=StopPatienceConfig(patience=20),
            batch_accumulation=4,
            warmup=WarmupConfig(n_steps=1000, function="linear"),
            no_learn_limit=10,
        ),
        data=hcp_1200_dataset_config.mean_std_file_ta_sa_00(hcp_1200_data_config.unp()),
        data_loaders=data_loader_config.batch_size_16(),
    )


### v/4
def brt_small_vp4_unp__00() -> ExperimentConfig:  # done
    return ExperimentConfig(
        description="lr: 1e-4, weight decay: 1e-2, warmup: 1000 steps, eta_min: 1e-6, batch size: 16, batch accum.: 4",
        model=model_config.brt_small_vp4_unp(),
        criterion=CrossEntropyLossConfig(),
        train=TrainConfig(
            optimizer=AdamWConfig(
                lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-2, exclude=["bias", "norm", "class_token", "pos_embed"]
            ),
            scheduler=CosineAnnealingLRConfig(type="epoch", t_max=100, eta_min=1e-6),
            stop_criterion=StopPatienceConfig(patience=20),
            batch_accumulation=4,
            warmup=WarmupConfig(n_steps=1000, function="linear"),
            no_learn_limit=10,
        ),
        data=hcp_1200_dataset_config.mean_std_file_ta(hcp_1200_data_config.unp()),
        data_loaders=data_loader_config.batch_size_16(),
    )


def brt_small_vp4_unp__01() -> ExperimentConfig:  # done
    return ExperimentConfig(
        description="lr: 1e-4, weight decay: 1e-2, warmup: 1000 steps, eta_min: 1e-6, batch size: 16, batch accum.: 4",
        model=model_config.brt_small_vp4_unp(),
        criterion=CrossEntropyLossConfig(),
        train=TrainConfig(
            optimizer=AdamWConfig(
                lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-2, exclude=["bias", "norm", "class_token", "pos_embed"]
            ),
            scheduler=CosineAnnealingLRConfig(type="epoch", t_max=100, eta_min=1e-6),
            stop_criterion=StopPatienceConfig(patience=20),
            batch_accumulation=4,
            warmup=WarmupConfig(n_steps=1000, function="linear"),
            no_learn_limit=10,
        ),
        data=hcp_1200_dataset_config.mean_std_file_ta_sa_00(hcp_1200_data_config.unp()),
        data_loaders=data_loader_config.batch_size_16(),
    )


def brt_small_vp4_unp__02() -> ExperimentConfig:  # done
    return ExperimentConfig(
        description="lr: 1e-4, weight decay: 1e-2, warmup: 1000 steps, eta_min: 1e-6, batch size: 16, batch accum.: 4",
        model=model_config.brt_small_vp4_unp(),
        criterion=CrossEntropyLossConfig(),
        train=TrainConfig(
            optimizer=AdamWConfig(
                lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-2, exclude=["bias", "norm", "class_token", "pos_embed"]
            ),
            scheduler=CosineAnnealingLRConfig(type="epoch", t_max=100, eta_min=1e-6),
            stop_criterion=StopPatienceConfig(patience=20),
            batch_accumulation=4,
            warmup=WarmupConfig(n_steps=1000, function="linear"),
            no_learn_limit=10,
        ),
        data=hcp_1200_dataset_config.mean_std_file_ta_sa_01(hcp_1200_data_config.unp()),
        data_loaders=data_loader_config.batch_size_16(),
    )


def brt_small_vc4_unp__00() -> ExperimentConfig:  # done
    return ExperimentConfig(
        description="lr: 1e-4, weight decay: 1e-2, warmup: 1000 steps, eta_min: 1e-6, batch size: 16, batch accum.: 4",
        model=model_config.brt_small_vc4_unp(),
        criterion=CrossEntropyLossConfig(),
        train=TrainConfig(
            optimizer=AdamWConfig(
                lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-2, exclude=["bias", "norm", "class_token", "pos_embed"]
            ),
            scheduler=CosineAnnealingLRConfig(type="epoch", t_max=100, eta_min=1e-6),
            stop_criterion=StopPatienceConfig(patience=20),
            batch_accumulation=4,
            warmup=WarmupConfig(n_steps=1000, function="linear"),
            no_learn_limit=10,
        ),
        data=hcp_1200_dataset_config.mean_std_file_ta(hcp_1200_data_config.unp()),
        data_loaders=data_loader_config.batch_size_16(),
    )


def brt_small_vc4_unp__01() -> ExperimentConfig:  # done
    return ExperimentConfig(
        description="lr: 1e-4, weight decay: 1e-2, warmup: 1000 steps, eta_min: 1e-6, batch size: 16, batch accum.: 4",
        model=model_config.brt_small_vc4_unp(),
        criterion=CrossEntropyLossConfig(),
        train=TrainConfig(
            optimizer=AdamWConfig(
                lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-2, exclude=["bias", "norm", "class_token", "pos_embed"]
            ),
            scheduler=CosineAnnealingLRConfig(type="epoch", t_max=100, eta_min=1e-6),
            stop_criterion=StopPatienceConfig(patience=20),
            batch_accumulation=4,
            warmup=WarmupConfig(n_steps=1000, function="linear"),
            no_learn_limit=10,
        ),
        data=hcp_1200_dataset_config.mean_std_file_ta_sa_00(hcp_1200_data_config.unp()),
        data_loaders=data_loader_config.batch_size_16(),
    )


# brt_medium
## prep
### t4
def brt_medium_t4p_prep__00() -> ExperimentConfig:  # done
    return ExperimentConfig(
        description="lr: 1e-4, weight decay: 1e-2, warmup: 1000 steps, eta_min: 1e-6, batch size: 16, batch accum.: 4",
        model=model_config.brt_medium_t4p_prep(),
        criterion=CrossEntropyLossConfig(),
        train=TrainConfig(
            optimizer=AdamWConfig(
                lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-2, exclude=["bias", "norm", "class_token", "pos_embed"]
            ),
            scheduler=CosineAnnealingLRConfig(type="epoch", t_max=100, eta_min=1e-6),
            stop_criterion=StopPatienceConfig(patience=20),
            batch_accumulation=4,
            warmup=WarmupConfig(n_steps=1000, function="linear"),
            no_learn_limit=10,
        ),
        data=hcp_1200_dataset_config.mean_std_voxel_file_ta(hcp_1200_data_config.prep()),
        data_loaders=data_loader_config.batch_size_16(),
    )


def brt_medium_t4c1_prep__00() -> ExperimentConfig:  # done
    return ExperimentConfig(
        description="lr: 1e-4, weight decay: 1e-2, warmup: 1000 steps, eta_min: 1e-6, batch size: 16, batch accum.: 4",
        model=model_config.brt_medium_t4c1_prep(),
        criterion=CrossEntropyLossConfig(),
        train=TrainConfig(
            optimizer=AdamWConfig(
                lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-2, exclude=["bias", "norm", "class_token", "pos_embed"]
            ),
            scheduler=CosineAnnealingLRConfig(type="epoch", t_max=100, eta_min=1e-6),
            stop_criterion=StopPatienceConfig(patience=20),
            batch_accumulation=4,
            warmup=WarmupConfig(n_steps=1000, function="linear"),
            no_learn_limit=10,
        ),
        data=hcp_1200_dataset_config.mean_std_voxel_file_ta(hcp_1200_data_config.prep()),
        data_loaders=data_loader_config.batch_size_16(),
    )


def brt_medium_t4c2_prep__00() -> ExperimentConfig:  # done, CV
    return ExperimentConfig(
        description="lr: 1e-4, weight decay: 1e-2, warmup: 1000 steps, eta_min: 1e-6, batch size: 16, batch accum.: 4",
        model=model_config.brt_medium_t4c2_prep(),
        criterion=CrossEntropyLossConfig(),
        train=TrainConfig(
            optimizer=AdamWConfig(
                lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-2, exclude=["bias", "norm", "class_token", "pos_embed"]
            ),
            scheduler=CosineAnnealingLRConfig(type="epoch", t_max=100, eta_min=1e-6),
            stop_criterion=StopPatienceConfig(patience=20),
            batch_accumulation=4,
            warmup=WarmupConfig(n_steps=1000, function="linear"),
            no_learn_limit=10,
        ),
        data=hcp_1200_dataset_config.mean_std_voxel_file_ta(hcp_1200_data_config.prep()),
        data_loaders=data_loader_config.batch_size_16(),
    )


### t8
def brt_medium_t8p_prep__00() -> ExperimentConfig:  # done
    return ExperimentConfig(
        description="lr: 1e-4, weight decay: 1e-2, warmup: 1000 steps, eta_min: 1e-6, batch size: 16, batch accum.: 4",
        model=model_config.brt_medium_t8p_prep(),
        criterion=CrossEntropyLossConfig(),
        train=TrainConfig(
            optimizer=AdamWConfig(
                lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-2, exclude=["bias", "norm", "class_token", "pos_embed"]
            ),
            scheduler=CosineAnnealingLRConfig(type="epoch", t_max=100, eta_min=1e-6),
            stop_criterion=StopPatienceConfig(patience=20),
            batch_accumulation=4,
            warmup=WarmupConfig(n_steps=1000, function="linear"),
            no_learn_limit=10,
        ),
        data=hcp_1200_dataset_config.mean_std_voxel_file_ta(hcp_1200_data_config.prep()),
        data_loaders=data_loader_config.batch_size_16(),
    )


def brt_medium_t8c1_prep__00() -> ExperimentConfig:  # done
    return ExperimentConfig(
        description="lr: 1e-4, weight decay: 1e-2, warmup: 1000 steps, eta_min: 1e-6, batch size: 16, batch accum.: 4",
        model=model_config.brt_medium_t8c1_prep(),
        criterion=CrossEntropyLossConfig(),
        train=TrainConfig(
            optimizer=AdamWConfig(
                lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-2, exclude=["bias", "norm", "class_token", "pos_embed"]
            ),
            scheduler=CosineAnnealingLRConfig(type="epoch", t_max=100, eta_min=1e-6),
            stop_criterion=StopPatienceConfig(patience=20),
            batch_accumulation=4,
            warmup=WarmupConfig(n_steps=1000, function="linear"),
            no_learn_limit=10,
        ),
        data=hcp_1200_dataset_config.mean_std_voxel_file_ta(hcp_1200_data_config.prep()),
        data_loaders=data_loader_config.batch_size_16(),
    )


def brt_medium_t8c2_prep__00() -> ExperimentConfig:  # done
    return ExperimentConfig(
        description="lr: 1e-4, weight decay: 1e-2, warmup: 1000 steps, eta_min: 1e-6, batch size: 16, batch accum.: 4",
        model=model_config.brt_medium_t8c2_prep(),
        criterion=CrossEntropyLossConfig(),
        train=TrainConfig(
            optimizer=AdamWConfig(
                lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-2, exclude=["bias", "norm", "class_token", "pos_embed"]
            ),
            scheduler=CosineAnnealingLRConfig(type="epoch", t_max=100, eta_min=1e-6),
            stop_criterion=StopPatienceConfig(patience=20),
            batch_accumulation=4,
            warmup=WarmupConfig(n_steps=1000, function="linear"),
            no_learn_limit=10,
        ),
        data=hcp_1200_dataset_config.mean_std_voxel_file_ta(hcp_1200_data_config.prep()),
        data_loaders=data_loader_config.batch_size_16(),
    )


### v/4
def brt_medium_vp4_prep__00() -> ExperimentConfig:  # done
    return ExperimentConfig(
        description="lr: 1e-4, weight decay: 1e-2, warmup: 1000 steps, eta_min: 1e-6, batch size: 16, batch accum.: 4",
        model=model_config.brt_medium_vp4_prep(),
        criterion=CrossEntropyLossConfig(),
        train=TrainConfig(
            optimizer=AdamWConfig(
                lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-2, exclude=["bias", "norm", "class_token", "pos_embed"]
            ),
            scheduler=CosineAnnealingLRConfig(type="epoch", t_max=100, eta_min=1e-6),
            stop_criterion=StopPatienceConfig(patience=20),
            batch_accumulation=4,
            warmup=WarmupConfig(n_steps=1000, function="linear"),
            no_learn_limit=10,
        ),
        data=hcp_1200_dataset_config.mean_std_voxel_file_ta(hcp_1200_data_config.prep()),
        data_loaders=data_loader_config.batch_size_16(),
    )


def brt_medium_vc4_prep__00() -> ExperimentConfig:  # done
    return ExperimentConfig(
        description="lr: 1e-4, weight decay: 1e-2, warmup: 1000 steps, eta_min: 1e-6, batch size: 16, batch accum.: 4",
        model=model_config.brt_medium_vc4_prep(),
        criterion=CrossEntropyLossConfig(),
        train=TrainConfig(
            optimizer=AdamWConfig(
                lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-2, exclude=["bias", "norm", "class_token", "pos_embed"]
            ),
            scheduler=CosineAnnealingLRConfig(type="epoch", t_max=100, eta_min=1e-6),
            stop_criterion=StopPatienceConfig(patience=20),
            batch_accumulation=4,
            warmup=WarmupConfig(n_steps=1000, function="linear"),
            no_learn_limit=10,
        ),
        data=hcp_1200_dataset_config.mean_std_voxel_file_ta(hcp_1200_data_config.prep()),
        data_loaders=data_loader_config.batch_size_16(),
    )


## unp
### t4
def brt_medium_t4p_unp__00() -> ExperimentConfig:
    return ExperimentConfig(
        description="lr: 1e-4, weight decay: 1e-2, warmup: 1000 steps, eta_min: 1e-6, batch size: 16, batch accum.: 4",
        model=model_config.brt_medium_t4p_unp(),
        criterion=CrossEntropyLossConfig(),
        train=TrainConfig(
            optimizer=AdamWConfig(
                lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-2, exclude=["bias", "norm", "class_token", "pos_embed"]
            ),
            scheduler=CosineAnnealingLRConfig(type="epoch", t_max=100, eta_min=1e-6),
            stop_criterion=StopPatienceConfig(patience=20),
            batch_accumulation=4,
            warmup=WarmupConfig(n_steps=1000, function="linear"),
            no_learn_limit=10,
        ),
        data=hcp_1200_dataset_config.mean_std_file_ta(hcp_1200_data_config.unp()),
        data_loaders=data_loader_config.batch_size_16(),
    )


def brt_medium_t4p_unp__01() -> ExperimentConfig:
    return ExperimentConfig(
        description="lr: 1e-4, weight decay: 1e-2, warmup: 1000 steps, eta_min: 1e-6, batch size: 16, batch accum.: 4",
        model=model_config.brt_medium_t4p_unp(),
        criterion=CrossEntropyLossConfig(),
        train=TrainConfig(
            optimizer=AdamWConfig(
                lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-2, exclude=["bias", "norm", "class_token", "pos_embed"]
            ),
            scheduler=CosineAnnealingLRConfig(type="epoch", t_max=100, eta_min=1e-6),
            stop_criterion=StopPatienceConfig(patience=20),
            batch_accumulation=4,
            warmup=WarmupConfig(n_steps=1000, function="linear"),
            no_learn_limit=10,
        ),
        data=hcp_1200_dataset_config.mean_std_file_ta_sa_00(hcp_1200_data_config.unp()),
        data_loaders=data_loader_config.batch_size_16(),
    )


def brt_medium_t4c1_unp__00() -> ExperimentConfig:
    return ExperimentConfig(
        description="lr: 1e-4, weight decay: 1e-2, warmup: 1000 steps, eta_min: 1e-6, batch size: 16, batch accum.: 4",
        model=model_config.brt_medium_t4c1_unp(),
        criterion=CrossEntropyLossConfig(),
        train=TrainConfig(
            optimizer=AdamWConfig(
                lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-2, exclude=["bias", "norm", "class_token", "pos_embed"]
            ),
            scheduler=CosineAnnealingLRConfig(type="epoch", t_max=100, eta_min=1e-6),
            stop_criterion=StopPatienceConfig(patience=20),
            batch_accumulation=4,
            warmup=WarmupConfig(n_steps=1000, function="linear"),
            no_learn_limit=10,
        ),
        data=hcp_1200_dataset_config.mean_std_file_ta(hcp_1200_data_config.unp()),
        data_loaders=data_loader_config.batch_size_16(),
    )


def brt_medium_t4c1_unp__01() -> ExperimentConfig:
    return ExperimentConfig(
        description="lr: 1e-4, weight decay: 1e-2, warmup: 1000 steps, eta_min: 1e-6, batch size: 16, batch accum.: 4",
        model=model_config.brt_medium_t4c1_unp(),
        criterion=CrossEntropyLossConfig(),
        train=TrainConfig(
            optimizer=AdamWConfig(
                lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-2, exclude=["bias", "norm", "class_token", "pos_embed"]
            ),
            scheduler=CosineAnnealingLRConfig(type="epoch", t_max=100, eta_min=1e-6),
            stop_criterion=StopPatienceConfig(patience=20),
            batch_accumulation=4,
            warmup=WarmupConfig(n_steps=1000, function="linear"),
            no_learn_limit=10,
        ),
        data=hcp_1200_dataset_config.mean_std_file_ta_sa_00(hcp_1200_data_config.unp()),
        data_loaders=data_loader_config.batch_size_16(),
    )


def brt_medium_t4c2_unp__00() -> ExperimentConfig:  # done
    return ExperimentConfig(
        description="lr: 1e-4, weight decay: 1e-2, warmup: 1000 steps, eta_min: 1e-6, batch size: 16, batch accum.: 4",
        model=model_config.brt_medium_t4c2_unp(),
        criterion=CrossEntropyLossConfig(),
        train=TrainConfig(
            optimizer=AdamWConfig(
                lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-2, exclude=["bias", "norm", "class_token", "pos_embed"]
            ),
            scheduler=CosineAnnealingLRConfig(type="epoch", t_max=100, eta_min=1e-6),
            stop_criterion=StopPatienceConfig(patience=20),
            batch_accumulation=4,
            warmup=WarmupConfig(n_steps=1000, function="linear"),
            no_learn_limit=10,
        ),
        data=hcp_1200_dataset_config.mean_std_file_ta(hcp_1200_data_config.unp()),
        data_loaders=data_loader_config.batch_size_16(),
    )


def brt_medium_t4c2_unp__01() -> ExperimentConfig:  # done
    return ExperimentConfig(
        description="lr: 1e-4, weight decay: 1e-2, warmup: 1000 steps, eta_min: 1e-6, batch size: 16, batch accum.: 4",
        model=model_config.brt_medium_t4c2_unp(),
        criterion=CrossEntropyLossConfig(),
        train=TrainConfig(
            optimizer=AdamWConfig(
                lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-2, exclude=["bias", "norm", "class_token", "pos_embed"]
            ),
            scheduler=CosineAnnealingLRConfig(type="epoch", t_max=100, eta_min=1e-6),
            stop_criterion=StopPatienceConfig(patience=20),
            batch_accumulation=4,
            warmup=WarmupConfig(n_steps=1000, function="linear"),
            no_learn_limit=10,
        ),
        data=hcp_1200_dataset_config.mean_std_file_ta_sa_00(hcp_1200_data_config.unp()),
        data_loaders=data_loader_config.batch_size_16(),
    )


def brt_medium_t4c2_unp__02() -> ExperimentConfig:  # done
    return ExperimentConfig(
        description="lr: 1e-4, weight decay: 1e-2, warmup: 1000 steps, eta_min: 1e-6, batch size: 16, batch accum.: 4",
        model=model_config.brt_medium_t4c2_unp(),
        criterion=CrossEntropyLossConfig(),
        train=TrainConfig(
            optimizer=AdamWConfig(
                lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-2, exclude=["bias", "norm", "class_token", "pos_embed"]
            ),
            scheduler=CosineAnnealingLRConfig(type="epoch", t_max=100, eta_min=1e-6),
            stop_criterion=StopPatienceConfig(patience=20),
            batch_accumulation=4,
            warmup=WarmupConfig(n_steps=1000, function="linear"),
            no_learn_limit=10,
        ),
        data=hcp_1200_dataset_config.mean_std_file_ta_sa_01(hcp_1200_data_config.unp()),
        data_loaders=data_loader_config.batch_size_16(),
    )


### t8
def brt_medium_t8p_unp__00() -> ExperimentConfig:
    return ExperimentConfig(
        description="lr: 1e-4, weight decay: 1e-2, warmup: 1000 steps, eta_min: 1e-6, batch size: 16, batch accum.: 4",
        model=model_config.brt_medium_t8p_unp(),
        criterion=CrossEntropyLossConfig(),
        train=TrainConfig(
            optimizer=AdamWConfig(
                lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-2, exclude=["bias", "norm", "class_token", "pos_embed"]
            ),
            scheduler=CosineAnnealingLRConfig(type="epoch", t_max=100, eta_min=1e-6),
            stop_criterion=StopPatienceConfig(patience=20),
            batch_accumulation=4,
            warmup=WarmupConfig(n_steps=1000, function="linear"),
            no_learn_limit=10,
        ),
        data=hcp_1200_dataset_config.mean_std_file_ta(hcp_1200_data_config.unp()),
        data_loaders=data_loader_config.batch_size_16(),
    )


def brt_medium_t8p_unp__01() -> ExperimentConfig:
    return ExperimentConfig(
        description="lr: 1e-4, weight decay: 1e-2, warmup: 1000 steps, eta_min: 1e-6, batch size: 16, batch accum.: 4",
        model=model_config.brt_medium_t8p_unp(),
        criterion=CrossEntropyLossConfig(),
        train=TrainConfig(
            optimizer=AdamWConfig(
                lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-2, exclude=["bias", "norm", "class_token", "pos_embed"]
            ),
            scheduler=CosineAnnealingLRConfig(type="epoch", t_max=100, eta_min=1e-6),
            stop_criterion=StopPatienceConfig(patience=20),
            batch_accumulation=4,
            warmup=WarmupConfig(n_steps=1000, function="linear"),
            no_learn_limit=10,
        ),
        data=hcp_1200_dataset_config.mean_std_file_ta_sa_00(hcp_1200_data_config.unp()),
        data_loaders=data_loader_config.batch_size_16(),
    )


def brt_medium_t8c1_unp__00() -> ExperimentConfig:
    return ExperimentConfig(
        description="lr: 1e-4, weight decay: 1e-2, warmup: 1000 steps, eta_min: 1e-6, batch size: 16, batch accum.: 4",
        model=model_config.brt_medium_t8c1_unp(),
        criterion=CrossEntropyLossConfig(),
        train=TrainConfig(
            optimizer=AdamWConfig(
                lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-2, exclude=["bias", "norm", "class_token", "pos_embed"]
            ),
            scheduler=CosineAnnealingLRConfig(type="epoch", t_max=100, eta_min=1e-6),
            stop_criterion=StopPatienceConfig(patience=20),
            batch_accumulation=4,
            warmup=WarmupConfig(n_steps=1000, function="linear"),
            no_learn_limit=10,
        ),
        data=hcp_1200_dataset_config.mean_std_file_ta(hcp_1200_data_config.unp()),
        data_loaders=data_loader_config.batch_size_16(),
    )


def brt_medium_t8c1_unp__01() -> ExperimentConfig:
    return ExperimentConfig(
        description="lr: 1e-4, weight decay: 1e-2, warmup: 1000 steps, eta_min: 1e-6, batch size: 16, batch accum.: 4",
        model=model_config.brt_medium_t8c1_unp(),
        criterion=CrossEntropyLossConfig(),
        train=TrainConfig(
            optimizer=AdamWConfig(
                lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-2, exclude=["bias", "norm", "class_token", "pos_embed"]
            ),
            scheduler=CosineAnnealingLRConfig(type="epoch", t_max=100, eta_min=1e-6),
            stop_criterion=StopPatienceConfig(patience=20),
            batch_accumulation=4,
            warmup=WarmupConfig(n_steps=1000, function="linear"),
            no_learn_limit=10,
        ),
        data=hcp_1200_dataset_config.mean_std_file_ta_sa_00(hcp_1200_data_config.unp()),
        data_loaders=data_loader_config.batch_size_16(),
    )


def brt_medium_t8c2_unp__00() -> ExperimentConfig:
    return ExperimentConfig(
        description="lr: 1e-4, weight decay: 1e-2, warmup: 1000 steps, eta_min: 1e-6, batch size: 16, batch accum.: 4",
        model=model_config.brt_medium_t8c2_unp(),
        criterion=CrossEntropyLossConfig(),
        train=TrainConfig(
            optimizer=AdamWConfig(
                lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-2, exclude=["bias", "norm", "class_token", "pos_embed"]
            ),
            scheduler=CosineAnnealingLRConfig(type="epoch", t_max=100, eta_min=1e-6),
            stop_criterion=StopPatienceConfig(patience=20),
            batch_accumulation=4,
            warmup=WarmupConfig(n_steps=1000, function="linear"),
            no_learn_limit=10,
        ),
        data=hcp_1200_dataset_config.mean_std_file_ta(hcp_1200_data_config.unp()),
        data_loaders=data_loader_config.batch_size_16(),
    )


def brt_medium_t8c2_unp__01() -> ExperimentConfig:
    return ExperimentConfig(
        description="lr: 1e-4, weight decay: 1e-2, warmup: 1000 steps, eta_min: 1e-6, batch size: 16, batch accum.: 4",
        model=model_config.brt_medium_t8c2_unp(),
        criterion=CrossEntropyLossConfig(),
        train=TrainConfig(
            optimizer=AdamWConfig(
                lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-2, exclude=["bias", "norm", "class_token", "pos_embed"]
            ),
            scheduler=CosineAnnealingLRConfig(type="epoch", t_max=100, eta_min=1e-6),
            stop_criterion=StopPatienceConfig(patience=20),
            batch_accumulation=4,
            warmup=WarmupConfig(n_steps=1000, function="linear"),
            no_learn_limit=10,
        ),
        data=hcp_1200_dataset_config.mean_std_file_ta_sa_00(hcp_1200_data_config.unp()),
        data_loaders=data_loader_config.batch_size_16(),
    )


### v/4
def brt_medium_vp4_unp__00() -> ExperimentConfig:
    return ExperimentConfig(
        description="lr: 1e-4, weight decay: 1e-2, warmup: 1000 steps, eta_min: 1e-6, batch size: 16, batch accum.: 4",
        model=model_config.brt_medium_vp4_unp(),
        criterion=CrossEntropyLossConfig(),
        train=TrainConfig(
            optimizer=AdamWConfig(
                lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-2, exclude=["bias", "norm", "class_token", "pos_embed"]
            ),
            scheduler=CosineAnnealingLRConfig(type="epoch", t_max=100, eta_min=1e-6),
            stop_criterion=StopPatienceConfig(patience=20),
            batch_accumulation=4,
            warmup=WarmupConfig(n_steps=1000, function="linear"),
            no_learn_limit=10,
        ),
        data=hcp_1200_dataset_config.mean_std_file_ta(hcp_1200_data_config.unp()),
        data_loaders=data_loader_config.batch_size_16(),
    )


def brt_medium_vp4_unp__01() -> ExperimentConfig:
    return ExperimentConfig(
        description="lr: 1e-4, weight decay: 1e-2, warmup: 1000 steps, eta_min: 1e-6, batch size: 16, batch accum.: 4",
        model=model_config.brt_medium_vp4_unp(),
        criterion=CrossEntropyLossConfig(),
        train=TrainConfig(
            optimizer=AdamWConfig(
                lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-2, exclude=["bias", "norm", "class_token", "pos_embed"]
            ),
            scheduler=CosineAnnealingLRConfig(type="epoch", t_max=100, eta_min=1e-6),
            stop_criterion=StopPatienceConfig(patience=20),
            batch_accumulation=4,
            warmup=WarmupConfig(n_steps=1000, function="linear"),
            no_learn_limit=10,
        ),
        data=hcp_1200_dataset_config.mean_std_file_ta_sa_00(hcp_1200_data_config.unp()),
        data_loaders=data_loader_config.batch_size_16(),
    )


def brt_medium_vc4_unp__00() -> ExperimentConfig:
    return ExperimentConfig(
        description="lr: 1e-4, weight decay: 1e-2, warmup: 1000 steps, eta_min: 1e-6, batch size: 16, batch accum.: 4",
        model=model_config.brt_medium_vc4_unp(),
        criterion=CrossEntropyLossConfig(),
        train=TrainConfig(
            optimizer=AdamWConfig(
                lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-2, exclude=["bias", "norm", "class_token", "pos_embed"]
            ),
            scheduler=CosineAnnealingLRConfig(type="epoch", t_max=100, eta_min=1e-6),
            stop_criterion=StopPatienceConfig(patience=20),
            batch_accumulation=4,
            warmup=WarmupConfig(n_steps=1000, function="linear"),
            no_learn_limit=10,
        ),
        data=hcp_1200_dataset_config.mean_std_file_ta(hcp_1200_data_config.unp()),
        data_loaders=data_loader_config.batch_size_16(),
    )


def brt_medium_vc4_unp__01() -> ExperimentConfig:
    return ExperimentConfig(
        description="lr: 1e-4, weight decay: 1e-2, warmup: 1000 steps, eta_min: 1e-6, batch size: 16, batch accum.: 4",
        model=model_config.brt_medium_vc4_unp(),
        criterion=CrossEntropyLossConfig(),
        train=TrainConfig(
            optimizer=AdamWConfig(
                lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-2, exclude=["bias", "norm", "class_token", "pos_embed"]
            ),
            scheduler=CosineAnnealingLRConfig(type="epoch", t_max=100, eta_min=1e-6),
            stop_criterion=StopPatienceConfig(patience=20),
            batch_accumulation=4,
            warmup=WarmupConfig(n_steps=1000, function="linear"),
            no_learn_limit=10,
        ),
        data=hcp_1200_dataset_config.mean_std_file_ta_sa_00(hcp_1200_data_config.unp()),
        data_loaders=data_loader_config.batch_size_16(),
    )


# brt_large
## prep
### t4
def brt_large_t4p_prep__00() -> ExperimentConfig:  # done
    return ExperimentConfig(
        description="lr: 1e-4, weight decay: 1e-2, warmup: 1000 steps, eta_min: 1e-6, batch size: 16, batch accum.: 4",
        model=model_config.brt_large_t4p_prep(),
        criterion=CrossEntropyLossConfig(),
        train=TrainConfig(
            optimizer=AdamWConfig(
                lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-2, exclude=["bias", "norm", "class_token", "pos_embed"]
            ),
            scheduler=CosineAnnealingLRConfig(type="epoch", t_max=100, eta_min=1e-6),
            stop_criterion=StopPatienceConfig(patience=20),
            batch_accumulation=4,
            warmup=WarmupConfig(n_steps=1000, function="linear"),
            no_learn_limit=10,
        ),
        data=hcp_1200_dataset_config.mean_std_voxel_file_ta(hcp_1200_data_config.prep()),
        data_loaders=data_loader_config.batch_size_16(),
    )


def brt_large_t4c1_prep__00() -> ExperimentConfig:  # done
    return ExperimentConfig(
        description="lr: 1e-4, weight decay: 1e-2, warmup: 1000 steps, eta_min: 1e-6, batch size: 16, batch accum.: 4",
        model=model_config.brt_large_t4c1_prep(),
        criterion=CrossEntropyLossConfig(),
        train=TrainConfig(
            optimizer=AdamWConfig(
                lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-2, exclude=["bias", "norm", "class_token", "pos_embed"]
            ),
            scheduler=CosineAnnealingLRConfig(type="epoch", t_max=100, eta_min=1e-6),
            stop_criterion=StopPatienceConfig(patience=20),
            batch_accumulation=4,
            warmup=WarmupConfig(n_steps=1000, function="linear"),
            no_learn_limit=10,
        ),
        data=hcp_1200_dataset_config.mean_std_voxel_file_ta(hcp_1200_data_config.prep()),
        data_loaders=data_loader_config.batch_size_16(),
    )


def brt_large_t4c2_prep__00() -> ExperimentConfig:  # done
    return ExperimentConfig(
        description="lr: 1e-4, weight decay: 1e-2, warmup: 1000 steps, eta_min: 1e-6, batch size: 16, batch accum.: 4",
        model=model_config.brt_large_t4c2_prep(),
        criterion=CrossEntropyLossConfig(),
        train=TrainConfig(
            optimizer=AdamWConfig(
                lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-2, exclude=["bias", "norm", "class_token", "pos_embed"]
            ),
            scheduler=CosineAnnealingLRConfig(type="epoch", t_max=100, eta_min=1e-6),
            stop_criterion=StopPatienceConfig(patience=20),
            batch_accumulation=4,
            warmup=WarmupConfig(n_steps=1000, function="linear"),
            no_learn_limit=10,
        ),
        data=hcp_1200_dataset_config.mean_std_voxel_file_ta(hcp_1200_data_config.prep()),
        data_loaders=data_loader_config.batch_size_16(),
    )


### t8
def brt_large_t8p_prep__00() -> ExperimentConfig:  # done
    return ExperimentConfig(
        description="lr: 1e-4, weight decay: 1e-2, warmup: 1000 steps, eta_min: 1e-6, batch size: 16, batch accum.: 4",
        model=model_config.brt_large_t8p_prep(),
        criterion=CrossEntropyLossConfig(),
        train=TrainConfig(
            optimizer=AdamWConfig(
                lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-2, exclude=["bias", "norm", "class_token", "pos_embed"]
            ),
            scheduler=CosineAnnealingLRConfig(type="epoch", t_max=100, eta_min=1e-6),
            stop_criterion=StopPatienceConfig(patience=20),
            batch_accumulation=4,
            warmup=WarmupConfig(n_steps=1000, function="linear"),
            no_learn_limit=10,
        ),
        data=hcp_1200_dataset_config.mean_std_voxel_file_ta(hcp_1200_data_config.prep()),
        data_loaders=data_loader_config.batch_size_16(),
    )


def brt_large_t8c1_prep__00() -> ExperimentConfig:  # done
    return ExperimentConfig(
        description="lr: 1e-4, weight decay: 1e-2, warmup: 1000 steps, eta_min: 1e-6, batch size: 16, batch accum.: 4",
        model=model_config.brt_large_t8c1_prep(),
        criterion=CrossEntropyLossConfig(),
        train=TrainConfig(
            optimizer=AdamWConfig(
                lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-2, exclude=["bias", "norm", "class_token", "pos_embed"]
            ),
            scheduler=CosineAnnealingLRConfig(type="epoch", t_max=100, eta_min=1e-6),
            stop_criterion=StopPatienceConfig(patience=20),
            batch_accumulation=4,
            warmup=WarmupConfig(n_steps=1000, function="linear"),
            no_learn_limit=10,
        ),
        data=hcp_1200_dataset_config.mean_std_voxel_file_ta(hcp_1200_data_config.prep()),
        data_loaders=data_loader_config.batch_size_16(),
    )


def brt_large_t8c2_prep__00() -> ExperimentConfig:  # done
    return ExperimentConfig(
        description="lr: 1e-4, weight decay: 1e-2, warmup: 1000 steps, eta_min: 1e-6, batch size: 16, batch accum.: 4",
        model=model_config.brt_large_t8c2_prep(),
        criterion=CrossEntropyLossConfig(),
        train=TrainConfig(
            optimizer=AdamWConfig(
                lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-2, exclude=["bias", "norm", "class_token", "pos_embed"]
            ),
            scheduler=CosineAnnealingLRConfig(type="epoch", t_max=100, eta_min=1e-6),
            stop_criterion=StopPatienceConfig(patience=20),
            batch_accumulation=4,
            warmup=WarmupConfig(n_steps=1000, function="linear"),
            no_learn_limit=10,
        ),
        data=hcp_1200_dataset_config.mean_std_voxel_file_ta(hcp_1200_data_config.prep()),
        data_loaders=data_loader_config.batch_size_16(),
    )


### v/4
def brt_large_vp4_prep__00() -> ExperimentConfig:  # done
    return ExperimentConfig(
        description="lr: 1e-4, weight decay: 1e-2, warmup: 1000 steps, eta_min: 1e-6, batch size: 16, batch accum.: 4",
        model=model_config.brt_large_vp4_prep(),
        criterion=CrossEntropyLossConfig(),
        train=TrainConfig(
            optimizer=AdamWConfig(
                lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-2, exclude=["bias", "norm", "class_token", "pos_embed"]
            ),
            scheduler=CosineAnnealingLRConfig(type="epoch", t_max=100, eta_min=1e-6),
            stop_criterion=StopPatienceConfig(patience=20),
            batch_accumulation=4,
            warmup=WarmupConfig(n_steps=1000, function="linear"),
            no_learn_limit=10,
        ),
        data=hcp_1200_dataset_config.mean_std_voxel_file_ta(hcp_1200_data_config.prep()),
        data_loaders=data_loader_config.batch_size_16(),
    )


def brt_large_vc4_prep__00() -> ExperimentConfig:  # done
    return ExperimentConfig(
        description="lr: 1e-4, weight decay: 1e-2, warmup: 1000 steps, eta_min: 1e-6, batch size: 16, batch accum.: 4",
        model=model_config.brt_large_vc4_prep(),
        criterion=CrossEntropyLossConfig(),
        train=TrainConfig(
            optimizer=AdamWConfig(
                lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-2, exclude=["bias", "norm", "class_token", "pos_embed"]
            ),
            scheduler=CosineAnnealingLRConfig(type="epoch", t_max=100, eta_min=1e-6),
            stop_criterion=StopPatienceConfig(patience=20),
            batch_accumulation=4,
            warmup=WarmupConfig(n_steps=1000, function="linear"),
            no_learn_limit=10,
        ),
        data=hcp_1200_dataset_config.mean_std_voxel_file_ta(hcp_1200_data_config.prep()),
        data_loaders=data_loader_config.batch_size_16(),
    )


## unp
### t4
def brt_large_t4p_unp__00() -> ExperimentConfig:
    return ExperimentConfig(
        description="lr: 1e-4, weight decay: 1e-2, warmup: 1000 steps, eta_min: 1e-6, batch size: 16, batch accum.: 4",
        model=model_config.brt_large_t4p_unp(),
        criterion=CrossEntropyLossConfig(),
        train=TrainConfig(
            optimizer=AdamWConfig(
                lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-2, exclude=["bias", "norm", "class_token", "pos_embed"]
            ),
            scheduler=CosineAnnealingLRConfig(type="epoch", t_max=100, eta_min=1e-6),
            stop_criterion=StopPatienceConfig(patience=20),
            batch_accumulation=4,
            warmup=WarmupConfig(n_steps=1000, function="linear"),
            no_learn_limit=10,
        ),
        data=hcp_1200_dataset_config.mean_std_file_ta(hcp_1200_data_config.unp()),
        data_loaders=data_loader_config.batch_size_16(),
    )


def brt_large_t4p_unp__01() -> ExperimentConfig:
    return ExperimentConfig(
        description="lr: 1e-4, weight decay: 1e-2, warmup: 1000 steps, eta_min: 1e-6, batch size: 16, batch accum.: 4",
        model=model_config.brt_large_t4p_unp(),
        criterion=CrossEntropyLossConfig(),
        train=TrainConfig(
            optimizer=AdamWConfig(
                lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-2, exclude=["bias", "norm", "class_token", "pos_embed"]
            ),
            scheduler=CosineAnnealingLRConfig(type="epoch", t_max=100, eta_min=1e-6),
            stop_criterion=StopPatienceConfig(patience=20),
            batch_accumulation=4,
            warmup=WarmupConfig(n_steps=1000, function="linear"),
            no_learn_limit=10,
        ),
        data=hcp_1200_dataset_config.mean_std_file_ta_sa_00(hcp_1200_data_config.unp()),
        data_loaders=data_loader_config.batch_size_16(),
    )


def brt_large_t4c1_unp__00() -> ExperimentConfig:
    return ExperimentConfig(
        description="lr: 1e-4, weight decay: 1e-2, warmup: 1000 steps, eta_min: 1e-6, batch size: 16, batch accum.: 4",
        model=model_config.brt_large_t4c1_unp(),
        criterion=CrossEntropyLossConfig(),
        train=TrainConfig(
            optimizer=AdamWConfig(
                lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-2, exclude=["bias", "norm", "class_token", "pos_embed"]
            ),
            scheduler=CosineAnnealingLRConfig(type="epoch", t_max=100, eta_min=1e-6),
            stop_criterion=StopPatienceConfig(patience=20),
            batch_accumulation=4,
            warmup=WarmupConfig(n_steps=1000, function="linear"),
            no_learn_limit=10,
        ),
        data=hcp_1200_dataset_config.mean_std_file_ta(hcp_1200_data_config.unp()),
        data_loaders=data_loader_config.batch_size_16(),
    )


def brt_large_t4c1_unp__01() -> ExperimentConfig:
    return ExperimentConfig(
        description="lr: 1e-4, weight decay: 1e-2, warmup: 1000 steps, eta_min: 1e-6, batch size: 16, batch accum.: 4",
        model=model_config.brt_large_t4c1_unp(),
        criterion=CrossEntropyLossConfig(),
        train=TrainConfig(
            optimizer=AdamWConfig(
                lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-2, exclude=["bias", "norm", "class_token", "pos_embed"]
            ),
            scheduler=CosineAnnealingLRConfig(type="epoch", t_max=100, eta_min=1e-6),
            stop_criterion=StopPatienceConfig(patience=20),
            batch_accumulation=4,
            warmup=WarmupConfig(n_steps=1000, function="linear"),
            no_learn_limit=10,
        ),
        data=hcp_1200_dataset_config.mean_std_file_ta_sa_00(hcp_1200_data_config.unp()),
        data_loaders=data_loader_config.batch_size_16(),
    )


def brt_large_t4c2_unp__00() -> ExperimentConfig:  # done
    return ExperimentConfig(
        description="lr: 1e-4, weight decay: 1e-2, warmup: 1000 steps, eta_min: 1e-6, batch size: 16, batch accum.: 4",
        model=model_config.brt_large_t4c2_unp(),
        criterion=CrossEntropyLossConfig(),
        train=TrainConfig(
            optimizer=AdamWConfig(
                lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-2, exclude=["bias", "norm", "class_token", "pos_embed"]
            ),
            scheduler=CosineAnnealingLRConfig(type="epoch", t_max=100, eta_min=1e-6),
            stop_criterion=StopPatienceConfig(patience=20),
            batch_accumulation=4,
            warmup=WarmupConfig(n_steps=1000, function="linear"),
            no_learn_limit=10,
        ),
        data=hcp_1200_dataset_config.mean_std_file_ta(hcp_1200_data_config.unp()),
        data_loaders=data_loader_config.batch_size_16(),
    )


def brt_large_t4c2_unp__01() -> ExperimentConfig:  # done
    return ExperimentConfig(
        description="lr: 1e-4, weight decay: 1e-2, warmup: 1000 steps, eta_min: 1e-6, batch size: 16, batch accum.: 4",
        model=model_config.brt_large_t4c2_unp(),
        criterion=CrossEntropyLossConfig(),
        train=TrainConfig(
            optimizer=AdamWConfig(
                lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-2, exclude=["bias", "norm", "class_token", "pos_embed"]
            ),
            scheduler=CosineAnnealingLRConfig(type="epoch", t_max=100, eta_min=1e-6),
            stop_criterion=StopPatienceConfig(patience=20),
            batch_accumulation=4,
            warmup=WarmupConfig(n_steps=1000, function="linear"),
            no_learn_limit=10,
        ),
        data=hcp_1200_dataset_config.mean_std_file_ta_sa_00(hcp_1200_data_config.unp()),
        data_loaders=data_loader_config.batch_size_16(),
    )


def brt_large_t4c2_unp__02() -> ExperimentConfig:  # done
    return ExperimentConfig(
        description="lr: 1e-4, weight decay: 1e-2, warmup: 1000 steps, eta_min: 1e-6, batch size: 16, batch accum.: 4",
        model=model_config.brt_large_t4c2_unp(),
        criterion=CrossEntropyLossConfig(),
        train=TrainConfig(
            optimizer=AdamWConfig(
                lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-2, exclude=["bias", "norm", "class_token", "pos_embed"]
            ),
            scheduler=CosineAnnealingLRConfig(type="epoch", t_max=100, eta_min=1e-6),
            stop_criterion=StopPatienceConfig(patience=20),
            batch_accumulation=4,
            warmup=WarmupConfig(n_steps=1000, function="linear"),
            no_learn_limit=10,
        ),
        data=hcp_1200_dataset_config.mean_std_file_ta_sa_01(hcp_1200_data_config.unp()),
        data_loaders=data_loader_config.batch_size_16(),
    )


### t8
def brt_large_t8p_unp__00() -> ExperimentConfig:
    return ExperimentConfig(
        description="lr: 1e-4, weight decay: 1e-2, warmup: 1000 steps, eta_min: 1e-6, batch size: 16, batch accum.: 4",
        model=model_config.brt_large_t8p_unp(),
        criterion=CrossEntropyLossConfig(),
        train=TrainConfig(
            optimizer=AdamWConfig(
                lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-2, exclude=["bias", "norm", "class_token", "pos_embed"]
            ),
            scheduler=CosineAnnealingLRConfig(type="epoch", t_max=100, eta_min=1e-6),
            stop_criterion=StopPatienceConfig(patience=20),
            batch_accumulation=4,
            warmup=WarmupConfig(n_steps=1000, function="linear"),
            no_learn_limit=10,
        ),
        data=hcp_1200_dataset_config.mean_std_file_ta(hcp_1200_data_config.unp()),
        data_loaders=data_loader_config.batch_size_16(),
    )


def brt_large_t8p_unp__01() -> ExperimentConfig:
    return ExperimentConfig(
        description="lr: 1e-4, weight decay: 1e-2, warmup: 1000 steps, eta_min: 1e-6, batch size: 16, batch accum.: 4",
        model=model_config.brt_large_t8p_unp(),
        criterion=CrossEntropyLossConfig(),
        train=TrainConfig(
            optimizer=AdamWConfig(
                lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-2, exclude=["bias", "norm", "class_token", "pos_embed"]
            ),
            scheduler=CosineAnnealingLRConfig(type="epoch", t_max=100, eta_min=1e-6),
            stop_criterion=StopPatienceConfig(patience=20),
            batch_accumulation=4,
            warmup=WarmupConfig(n_steps=1000, function="linear"),
            no_learn_limit=10,
        ),
        data=hcp_1200_dataset_config.mean_std_file_ta_sa_00(hcp_1200_data_config.unp()),
        data_loaders=data_loader_config.batch_size_16(),
    )


def brt_large_t8c1_unp__00() -> ExperimentConfig:
    return ExperimentConfig(
        description="lr: 1e-4, weight decay: 1e-2, warmup: 1000 steps, eta_min: 1e-6, batch size: 16, batch accum.: 4",
        model=model_config.brt_large_t8c1_unp(),
        criterion=CrossEntropyLossConfig(),
        train=TrainConfig(
            optimizer=AdamWConfig(
                lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-2, exclude=["bias", "norm", "class_token", "pos_embed"]
            ),
            scheduler=CosineAnnealingLRConfig(type="epoch", t_max=100, eta_min=1e-6),
            stop_criterion=StopPatienceConfig(patience=20),
            batch_accumulation=4,
            warmup=WarmupConfig(n_steps=1000, function="linear"),
            no_learn_limit=10,
        ),
        data=hcp_1200_dataset_config.mean_std_file_ta(hcp_1200_data_config.unp()),
        data_loaders=data_loader_config.batch_size_16(),
    )


def brt_large_t8c1_unp__01() -> ExperimentConfig:
    return ExperimentConfig(
        description="lr: 1e-4, weight decay: 1e-2, warmup: 1000 steps, eta_min: 1e-6, batch size: 16, batch accum.: 4",
        model=model_config.brt_large_t8c1_unp(),
        criterion=CrossEntropyLossConfig(),
        train=TrainConfig(
            optimizer=AdamWConfig(
                lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-2, exclude=["bias", "norm", "class_token", "pos_embed"]
            ),
            scheduler=CosineAnnealingLRConfig(type="epoch", t_max=100, eta_min=1e-6),
            stop_criterion=StopPatienceConfig(patience=20),
            batch_accumulation=4,
            warmup=WarmupConfig(n_steps=1000, function="linear"),
            no_learn_limit=10,
        ),
        data=hcp_1200_dataset_config.mean_std_file_ta_sa_00(hcp_1200_data_config.unp()),
        data_loaders=data_loader_config.batch_size_16(),
    )


def brt_large_t8c2_unp__00() -> ExperimentConfig:
    return ExperimentConfig(
        description="lr: 1e-4, weight decay: 1e-2, warmup: 1000 steps, eta_min: 1e-6, batch size: 16, batch accum.: 4",
        model=model_config.brt_large_t8c2_unp(),
        criterion=CrossEntropyLossConfig(),
        train=TrainConfig(
            optimizer=AdamWConfig(
                lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-2, exclude=["bias", "norm", "class_token", "pos_embed"]
            ),
            scheduler=CosineAnnealingLRConfig(type="epoch", t_max=100, eta_min=1e-6),
            stop_criterion=StopPatienceConfig(patience=20),
            batch_accumulation=4,
            warmup=WarmupConfig(n_steps=1000, function="linear"),
            no_learn_limit=10,
        ),
        data=hcp_1200_dataset_config.mean_std_file_ta(hcp_1200_data_config.unp()),
        data_loaders=data_loader_config.batch_size_16(),
    )


def brt_large_t8c2_unp__01() -> ExperimentConfig:
    return ExperimentConfig(
        description="lr: 1e-4, weight decay: 1e-2, warmup: 1000 steps, eta_min: 1e-6, batch size: 16, batch accum.: 4",
        model=model_config.brt_large_t8c2_unp(),
        criterion=CrossEntropyLossConfig(),
        train=TrainConfig(
            optimizer=AdamWConfig(
                lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-2, exclude=["bias", "norm", "class_token", "pos_embed"]
            ),
            scheduler=CosineAnnealingLRConfig(type="epoch", t_max=100, eta_min=1e-6),
            stop_criterion=StopPatienceConfig(patience=20),
            batch_accumulation=4,
            warmup=WarmupConfig(n_steps=1000, function="linear"),
            no_learn_limit=10,
        ),
        data=hcp_1200_dataset_config.mean_std_file_ta_sa_00(hcp_1200_data_config.unp()),
        data_loaders=data_loader_config.batch_size_16(),
    )


### v/4
def brt_large_vp4_unp__00() -> ExperimentConfig:
    return ExperimentConfig(
        description="lr: 1e-4, weight decay: 1e-2, warmup: 1000 steps, eta_min: 1e-6, batch size: 16, batch accum.: 4",
        model=model_config.brt_large_vp4_unp(),
        criterion=CrossEntropyLossConfig(),
        train=TrainConfig(
            optimizer=AdamWConfig(
                lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-2, exclude=["bias", "norm", "class_token", "pos_embed"]
            ),
            scheduler=CosineAnnealingLRConfig(type="epoch", t_max=100, eta_min=1e-6),
            stop_criterion=StopPatienceConfig(patience=20),
            batch_accumulation=4,
            warmup=WarmupConfig(n_steps=1000, function="linear"),
            no_learn_limit=10,
        ),
        data=hcp_1200_dataset_config.mean_std_file_ta(hcp_1200_data_config.unp()),
        data_loaders=data_loader_config.batch_size_16(),
    )


def brt_large_vp4_unp__01() -> ExperimentConfig:
    return ExperimentConfig(
        description="lr: 1e-4, weight decay: 1e-2, warmup: 1000 steps, eta_min: 1e-6, batch size: 16, batch accum.: 4",
        model=model_config.brt_large_vp4_unp(),
        criterion=CrossEntropyLossConfig(),
        train=TrainConfig(
            optimizer=AdamWConfig(
                lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-2, exclude=["bias", "norm", "class_token", "pos_embed"]
            ),
            scheduler=CosineAnnealingLRConfig(type="epoch", t_max=100, eta_min=1e-6),
            stop_criterion=StopPatienceConfig(patience=20),
            batch_accumulation=4,
            warmup=WarmupConfig(n_steps=1000, function="linear"),
            no_learn_limit=10,
        ),
        data=hcp_1200_dataset_config.mean_std_file_ta_sa_00(hcp_1200_data_config.unp()),
        data_loaders=data_loader_config.batch_size_16(),
    )


def brt_large_vc4_unp__00() -> ExperimentConfig:
    return ExperimentConfig(
        description="lr: 1e-4, weight decay: 1e-2, warmup: 1000 steps, eta_min: 1e-6, batch size: 16, batch accum.: 4",
        model=model_config.brt_large_vc4_unp(),
        criterion=CrossEntropyLossConfig(),
        train=TrainConfig(
            optimizer=AdamWConfig(
                lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-2, exclude=["bias", "norm", "class_token", "pos_embed"]
            ),
            scheduler=CosineAnnealingLRConfig(type="epoch", t_max=100, eta_min=1e-6),
            stop_criterion=StopPatienceConfig(patience=20),
            batch_accumulation=4,
            warmup=WarmupConfig(n_steps=1000, function="linear"),
            no_learn_limit=10,
        ),
        data=hcp_1200_dataset_config.mean_std_file_ta(hcp_1200_data_config.unp()),
        data_loaders=data_loader_config.batch_size_16(),
    )


def brt_large_vc4_unp__01() -> ExperimentConfig:
    return ExperimentConfig(
        description="lr: 1e-4, weight decay: 1e-2, warmup: 1000 steps, eta_min: 1e-6, batch size: 16, batch accum.: 4",
        model=model_config.brt_large_vc4_unp(),
        criterion=CrossEntropyLossConfig(),
        train=TrainConfig(
            optimizer=AdamWConfig(
                lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-2, exclude=["bias", "norm", "class_token", "pos_embed"]
            ),
            scheduler=CosineAnnealingLRConfig(type="epoch", t_max=100, eta_min=1e-6),
            stop_criterion=StopPatienceConfig(patience=20),
            batch_accumulation=4,
            warmup=WarmupConfig(n_steps=1000, function="linear"),
            no_learn_limit=10,
        ),
        data=hcp_1200_dataset_config.mean_std_file_ta_sa_00(hcp_1200_data_config.unp()),
        data_loaders=data_loader_config.batch_size_16(),
    )
