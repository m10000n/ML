import argparse
from pathlib import Path

from config.config import DATASET, MODEL
from config.transfer import REMOTE_BASE_DIR

LOCAL_BASE_DIR = Path(__file__).parent.parent
REMOTE_BASE_DIR = Path(REMOTE_BASE_DIR)

MODEL_PATH = Path("model") / MODEL
LOCAL_MODEL_PATH = LOCAL_BASE_DIR / MODEL_PATH
REMOTE_MODEL_PATH = REMOTE_BASE_DIR / MODEL_PATH

DATALOADER_PATH = Path("data")
LOCAL_DATALOADER_PATH = LOCAL_BASE_DIR / DATALOADER_PATH
REMOTE_DATALOADER_PATH = REMOTE_BASE_DIR / DATALOADER_PATH

DATASET_PATH = DATALOADER_PATH / DATASET
LOCAL_DATASET_PATH = LOCAL_BASE_DIR / DATASET_PATH
REMOTE_DATASET_PATH = REMOTE_BASE_DIR / DATASET_PATH

CONFIG_PATH = Path("config")
LOCAL_CONFIG_PATH = LOCAL_BASE_DIR / CONFIG_PATH
REMOTE_CONFIG_PATH = REMOTE_BASE_DIR / CONFIG_PATH

HELPER_PATH = Path("helper")
LOCAL_HELPER_PATH = LOCAL_BASE_DIR / HELPER_PATH
REMOTE_HELPER_PATH = REMOTE_BASE_DIR / HELPER_PATH

SHELL_PATH = HELPER_PATH / "shell"
LOCAL_SHELL_PATH = LOCAL_BASE_DIR / SHELL_PATH
REMOTE_SHELL_PATH = REMOTE_BASE_DIR / SHELL_PATH

ENV_PATH = HELPER_PATH / "env"
LOCAL_ENV_PATH = LOCAL_BASE_DIR / ENV_PATH
REMOTE_ENV_PATH = REMOTE_BASE_DIR / ENV_PATH

TEMP_PATH = Path("temp")
LOCAL_TEMP_PATH = LOCAL_BASE_DIR / TEMP_PATH
REMOTE_TEMP_PATH = REMOTE_BASE_DIR / TEMP_PATH


def get_file_paths_in_folder(folder):
    return [file for file in Path(folder).glob("*") if file.is_file()]


if __name__ == "__main__":
    paths = {
        "local_base_dir": LOCAL_BASE_DIR,
        "remote_base_dir": REMOTE_BASE_DIR,
        "model": MODEL_PATH,
        "local_model": LOCAL_MODEL_PATH,
        "remote_model": REMOTE_MODEL_PATH,
        "dataloader": DATALOADER_PATH,
        "local_dataloader": LOCAL_DATALOADER_PATH,
        "remote_dataloader": REMOTE_DATALOADER_PATH,
        "dataset": DATASET_PATH,
        "local_dataset": LOCAL_DATASET_PATH,
        "remote_dataset": REMOTE_DATASET_PATH,
        "config": CONFIG_PATH,
        "local_config": LOCAL_CONFIG_PATH,
        "remote_config": REMOTE_CONFIG_PATH,
        "helper": HELPER_PATH,
        "local_helper": LOCAL_HELPER_PATH,
        "remote_helper": REMOTE_HELPER_PATH,
        "shell": SHELL_PATH,
        "local_shell": LOCAL_SHELL_PATH,
        "remote_shell": REMOTE_SHELL_PATH,
        "env": ENV_PATH,
        "local_env": LOCAL_ENV_PATH,
        "remote_env": REMOTE_ENV_PATH,
        "temp": TEMP_PATH,
        "local_temp": LOCAL_TEMP_PATH,
        "remote_temp": REMOTE_TEMP_PATH,
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("path", choices=list(paths.keys()))
    parser.add_argument("-m", action="store_true")
    args = parser.parse_args()

    path = paths[args.path]

    if args.m:
        if args.path.startswith(("local", "remote")):
            raise SystemExit(
                "It does not make sense to retrieve a module path for an absolute path."
            )
        print(str(path).replace("/", "."))
    else:
        print(path)
