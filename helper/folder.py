import os.path

from helper.path import RESULT_PATH_A


def create_result(exp_name):
    model_result_path = RESULT_PATH_A / exp_name

    os.makedirs(name=RESULT_PATH_A, exist_ok=True)
    os.makedirs(name=model_result_path, exist_ok=True)

    if has_content(model_result_path):
        raise FileExistsError("There are already files in the models results folder.")

    os.makedirs(name=model_result_path / "files", exist_ok=True)


def has_content(path):
    return path.exists() and any(path.iterdir())


def get_folders(path):
    return [item for item in path.iterdir() if item.is_dir()]


def get_files(path):
    return [item for item in path.iterdir() if item.is_file()]
