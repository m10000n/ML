from data import data_loader_config
from data.dummy_data import get_dummy_datasets_config
from model.criterion import CrossEntropyLossConfig
from model.experiment import ExperimentConfig, TrainConfig
from model.He_ResNet.architecture import model_config
from model.optimizer import SGDConfig
from model.scheduler import ReduceLROnPlateauConfig
from model.stop_criterion import StopEpochConfig


def resnet_34_a__dummy() -> ExperimentConfig:
    return ExperimentConfig(
        description="original hyperparameters",
        model=model_config.resnet_34_a(),
        criterion=CrossEntropyLossConfig(),
        train=TrainConfig(
            optimizer=SGDConfig(lr=0.1, momentum=0.9, weight_decay=0.0001),
            # paper: lr is reduced when the error plateaus
            scheduler=ReduceLROnPlateauConfig(mode="min", factor=0.1, patience=10),
            # paper: trained for up to 600.000 iterations, batch size: 256, images: 1.28 mio => 5000 iterations / epoch
            # => ~ 120 epochs
            stop_criterion=StopEpochConfig(n_epochs=120),
        ),
        data=get_dummy_datasets_config(n_classes=10, n_samples=100, shape=(3, 224, 224)),
        data_loaders=data_loader_config.batch_size_256(),
    )
