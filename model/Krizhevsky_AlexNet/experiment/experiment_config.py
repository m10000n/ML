from data import data_loader_config
from data.dummy_data import get_dummy_datasets_config
from model.criterion import CrossEntropyLossConfig
from model.experiment import ExperimentConfig, TrainConfig
from model.Krizhevsky_AlexNet.architecture import model_config
from model.optimizer import SGDConfig
from model.scheduler import ReduceLROnPlateauConfig
from model.stop_criterion import StopEpochConfig


def alex_net__dummy() -> ExperimentConfig:
    return ExperimentConfig(
        description="original hyperparameters",
        criterion=CrossEntropyLossConfig(),
        model=model_config.alex_net(),
        train=TrainConfig(
            optimizer=SGDConfig(lr=0.01, momentum=0.9, weight_decay=0.0005),
            # paper: manual lr adjustment when loss stopped improving
            scheduler=ReduceLROnPlateauConfig(mode="min", factor=0.1, patience=10),
            # paper: roughly 90 epochs
            stop_criterion=StopEpochConfig(n_epochs=90),
        ),
        data=get_dummy_datasets_config(n_classes=10, n_samples=100, shape=(3, 224, 224)),
        data_loaders=data_loader_config.batch_size_128(),
    )
