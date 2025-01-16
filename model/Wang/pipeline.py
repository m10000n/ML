import inspect
from pathlib import Path

import torch.cuda

import data.hcp_openacces.dataloader as dataloader
from config.exp_file_names import (
    DATALOADER_FILE_NAME,
    DATASET_FILE_NAME,
    DEPENDENCY_FILE_NAME,
    LOG_FILE_NAME,
    MODEL_FILE_NAME,
    MODEL_HELPER_FILE_NAME,
    PIPELINE_FILE_NAME,
    TEST_FILE_NAME,
    TRAIN_FILE_NAME,
    TRAINED_MODEL_FILE_NAME,
)
from data.hcp_openacces.dataset import TaskDataset as Dataset
from helper import model as model_helper
from helper.env import save_dependencies
from helper.folder import create_result
from helper.log import Log
from helper.model import save_exp_files
from helper.path import RESULT_PATH_A
from helper.plot import plot_all
from helper.system import get_num_physical_cores

# import the model you want to use
from model.Wang.model import Model as Model
from model.Wang.residual import Residual
from model.Wang.shortcut import Shortcut
from model.Wang.test import test as test
from model.Wang.train import train as train

EXP_NAME = "original"
NUM_WORKERS = max(1, get_num_physical_cores() - 1)

EXP_RESULT_FOLDER = RESULT_PATH_A / EXP_NAME
EXP_FILE_FOLDER = EXP_RESULT_FOLDER / "files"

EXP_FILES = [
    (
        inspect.getfile(Model),
        MODEL_FILE_NAME.format(exp_name=EXP_NAME),
    ),
    (inspect.getfile(Residual), f"{EXP_NAME}__residual_block.py"),
    (inspect.getfile(Shortcut), f"{EXP_NAME}__shortcut_block.py"),
    (Path(__file__), PIPELINE_FILE_NAME.format(exp_name=EXP_NAME)),
    (
        inspect.getfile(train),
        TRAIN_FILE_NAME.format(exp_name=EXP_NAME),
    ),
    (inspect.getfile(test), TEST_FILE_NAME.format(exp_name=EXP_NAME)),
    (
        inspect.getfile(dataloader),
        DATALOADER_FILE_NAME.format(exp_name=EXP_NAME),
    ),
    (
        inspect.getfile(Dataset),
        DATASET_FILE_NAME.format(exp_name=EXP_NAME),
    ),
    (
        inspect.getfile(model_helper),
        MODEL_HELPER_FILE_NAME.format(exp_name=EXP_NAME),
    ),
]


def main():
    create_result(exp_name=EXP_NAME)

    save_dependencies(
        path=EXP_RESULT_FOLDER / DEPENDENCY_FILE_NAME.format(exp_name=EXP_NAME)
    )
    save_exp_files(exp_file_folder=EXP_FILE_FOLDER, files=EXP_FILES)

    log = Log(file_path=EXP_RESULT_FOLDER / LOG_FILE_NAME.format(exp_name=EXP_NAME))

    if not Dataset.fully_downloaded():
        Dataset.download_dataset(num_workers=NUM_WORKERS)

    ids = dataloader.get_ids(ratios=(0.7, 0.1, 0.2))
    log.set_data_ids(ids)
    log.update()

    datasets = {
        "train": Dataset(ids["train"], is_train=True),
        "val": Dataset(ids["val"]),
    }
    test_dataset = Dataset(ids["test"])

    world_size = torch.cuda.device_count()

    trained_model_file = EXP_RESULT_FOLDER / TRAINED_MODEL_FILE_NAME.format(
        exp_name=EXP_NAME
    )

    if world_size <= 1:
        train(
            rank=0,
            world_size=world_size,
            model_class=Model,
            datasets=datasets,
            num_workers=NUM_WORKERS,
            model_path=trained_model_file,
            log=log,
        )
        test(
            rank=0,
            world_size=world_size,
            model_class=Model,
            dataset=test_dataset,
            num_workers=NUM_WORKERS,
            model_path=trained_model_file,
            log=log,
        )
    else:
        torch.multiprocessing.spawn(
            fn=train,
            args=(world_size, Model, datasets, NUM_WORKERS, trained_model_file, log),
            nprocs=world_size,
        )
        torch.multiprocessing.spawn(
            fn=test,
            args=(
                world_size,
                Model,
                test_dataset,
                NUM_WORKERS,
                trained_model_file,
                log,
            ),
            nprocs=world_size,
        )

    log.end()
    plot_all(exp_name=EXP_NAME, log=log, class_names=Dataset.TASKS)


if __name__ == "__main__":
    main()
