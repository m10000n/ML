import sys
from re import Pattern
from types import FunctionType
from typing import List, Tuple

from config import project
from helper import function, path
from helper.class_ import Data_, DataConfig_
from helper.print import print_error, print_info
from helper.tmux import in_tmux

_DATA_CONFIG_PATH = path.dataset(absolute=True) / f"{project.get_dataset().lower()}_data_config.py"


def __new(name: str) -> None:
    name = name.lower()
    dataset_dir = path.data(absolute=True) / name

    if dataset_dir.exists():
        print_error(f"Dataset `{name}` already exists.")
        sys.exit(1)

    dataset_dir.mkdir(parents=False, exist_ok=False)
    (dataset_dir / "__init__.py").touch(exist_ok=False)
    (dataset_dir / f"{name}_data.py").touch(exist_ok=False)
    (dataset_dir / f"{name}_data_config.py").touch(exist_ok=False)
    (dataset_dir / f"{name}_dataset.py").touch(exist_ok=False)
    (dataset_dir / f"{name}_dataset_config.py").touch(exist_ok=False)

    print_info(f"Created dataset structure.")


def __available_datasets(details: bool = False, pattern: str | Pattern = "") -> None:
    dataset_name = project.get_dataset()
    try:
        functions = _get_data_config_f(pattern=pattern)
    except ModuleNotFoundError as e:
        print_error(f"Failed to display available datasets for `{dataset_name}`. {e.args[0]}.")
        sys.exit(1)

    if functions:
        print(f"Available datasets for `{dataset_name}`:")
        for name, func in functions:
            config: DataConfig_ = func()
            if details:
                print(f"\t{name} - {config.name + f', {config.description}' if config.description else ""}")
            else:
                print(f"\t{name}")
    else:
        print_info(
            f"No datasets found for `{dataset_name}`. Define a dataset configuration function in `{path.make_relative(_DATA_CONFIG_PATH)}`."
        )


def __size_info(dataset_name: str) -> None:
    error_message = f"Failed to display size information for `{dataset_name} - {project.get_model()}`. "
    try:
        functions = _get_data_config_f()
    except ModuleNotFoundError as e:
        print_error(f"{error_message} {e.args[0]}.")
        sys.exit(1)

    for name, func in functions:
        if dataset_name == name:
            config: DataConfig_ = func()
            data: Data_ = Data_.create(config)
            data.size_info()
            return

    print_error(f"{error_message} Dataset not found.")
    sys.exit(1)


@in_tmux(window_name="data")
def __download(dataset_name: str) -> None:
    error_message = f"Failed to download `{dataset_name}` for `{project.get_dataset()}`. "
    try:
        functions = _get_data_config_f()
    except ModuleNotFoundError as e:
        print_error(f"{error_message} {e.args[0]}.")
        sys.exit(1)

    for name, func in functions:
        if dataset_name == name:
            config: DataConfig_ = func()
            data: Data_ = Data_.create(config)
            data.download()
            return

    print_error(f"{error_message} Dataset not found.")
    sys.exit(1)


def _get_data_config_f(pattern: str | Pattern = "") -> List[Tuple[str, FunctionType]]:
    try:
        return function._get_f(file_path=_DATA_CONFIG_PATH, pattern=pattern)
    except ModuleNotFoundError:
        raise ModuleNotFoundError(f"Dataset configuration file ({path.make_relative(_DATA_CONFIG_PATH)}) not found.")
