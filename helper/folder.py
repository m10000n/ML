import os.path


def create_results(model_path, model_name):
    results_path = model_path / "results"
    model_path = results_path / model_name

    os.makedirs(name=results_path, exist_ok=True)
    os.makedirs(name=model_path, exist_ok=True)

    if has_content(model_path):
        raise FileExistsError("There are already files in the models results folder.")


def has_content(path):
    return path.exists() and any(path.iterdir())


def get_folders(path):
    return [item for item in path.iterdir() if item.is_dir()]


def get_files(path):
    return [item for item in path.iterdir() if item.is_file()]
