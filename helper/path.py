import argparse
from pathlib import Path

from config.config import DATASET, MODEL
from config.transfer import REMOTE_BASE_DIR

BASE_DIR = Path(__file__).parent.parent
BASE_DIR_REMOTE = Path(REMOTE_BASE_DIR)

MODEL_PATH_R = Path("model") / MODEL
MODEL_PATH_A = BASE_DIR / MODEL_PATH_R
MODEL_PATH_REMOTE = BASE_DIR_REMOTE / MODEL_PATH_R

RESULT_PATH_R = MODEL_PATH_R / "result"
RESULT_PATH_A = BASE_DIR / RESULT_PATH_R
RESULT_PATH_REMOTE = BASE_DIR_REMOTE / RESULT_PATH_R

DATALOADER_PATH_R = Path("data")
DATALOADER_PATH_A = BASE_DIR / DATALOADER_PATH_R
DATALOADER_PATH_REMOTE = BASE_DIR_REMOTE / DATALOADER_PATH_R

DATASET_PATH_R = DATALOADER_PATH_R / DATASET
DATASET_PATH_A = BASE_DIR / DATASET_PATH_R
DATASET_PATH_REMOTE = BASE_DIR_REMOTE / DATASET_PATH_R

CONFIG_PATH_R = Path("config")
CONFIG_PATH_A = BASE_DIR / CONFIG_PATH_R
CONFIG_PATH_REMOTE = BASE_DIR_REMOTE / CONFIG_PATH_R

HELPER_PATH_R = Path("helper")
HELPER_PATH_A = BASE_DIR / HELPER_PATH_R
HELPER_PATH_REMOTE = BASE_DIR_REMOTE / HELPER_PATH_R

SHELL_PATH_R = HELPER_PATH_R / "shell"
SHELL_PATH_A = BASE_DIR / SHELL_PATH_R
SHELL_PATH_REMOTE = BASE_DIR_REMOTE / SHELL_PATH_R

ENV_PATH_R = HELPER_PATH_R / "env"
ENV_PATH_A = BASE_DIR / ENV_PATH_R
ENV_PATH_REMOTE = BASE_DIR_REMOTE / ENV_PATH_R


def get_file_paths_in_folder(folder):
    return [file for file in Path(folder).glob("*") if file.is_file()]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        choices=[
            "base_dir",
            "model",
            "result",
            "dataloader",
            "dataset",
            "config",
            "helper",
            "shell",
            "env",
        ],
    )
    parser.add_argument("option", choices=["a", "r", "rmt", "m"])
    args = parser.parse_args()

    if args.path == "base_dir":
        if args.option == "a":
            print(BASE_DIR)
        elif args.option == "r":
            print("./")
        elif args.option == "rmt":
            print(BASE_DIR_REMOTE)
        elif args.option == "m":
            print("")
    else:
        path = args.path.upper() + "_PATH"
        if args.option == "a":
            print(globals()[path + "_A"])
        elif args.option == "r":
            print(globals()[path + "_R"])
        elif args.option == "rmt":
            print(globals()[path + "_REMOTE"])
        elif args.option == "m":
            print(str(globals()[path + "_R"]).replace("/", "."))
