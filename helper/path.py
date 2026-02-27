# this file is used before the python environment is ready

import inspect
from pathlib import Path
from typing import List, Optional, Union

from config import project
from helper.validator import validator_


# getters for dirs in the project
def home() -> Path:
    return Path.home()


def project_root() -> Path:
    return Path(__file__).parent.parent


def config(absolute: bool = False) -> Path:
    path = Path("config")
    return make_absolute(path) if absolute else path


def model(absolute: bool = False) -> Path:
    path = Path("model")
    return make_absolute(path) if absolute else path


def model_impl(model_name: Optional[str] = None, absolute: bool = False) -> Path:
    model_name_ = project.get_model() if model_name is None else model_name
    path = model(absolute=True) / model_name_

    if not path.exists():
        raise ValueError(f"`{model_name_}` is not a valid model.")

    return path if absolute else make_relative(path)


def architecture(model_name: Optional[str] = None, absolute: bool = False) -> Path:
    return model_impl(model_name=model_name, absolute=absolute) / "architecture"


def experiment(model_name: Optional[str] = None, absolute: bool = False) -> Path:
    return model_impl(model_name=model_name, absolute=absolute) / "experiment"


def cross_validation(model_name: Optional[str] = None, absolute: bool = False) -> Path:
    return experiment(model_name=model_name, absolute=absolute) / "CV"


def data(absolute: bool = False) -> Path:
    path = Path("data")
    return make_absolute(path) if absolute else path


def dataset(absolute: bool = False) -> Path:
    return data(absolute) / project.get_dataset()


def helper(absolute: bool = False) -> Path:
    path = Path("helper")
    return make_absolute(path) if absolute else path


def plot(absolute: bool = False) -> Path:
    path = Path("plot")
    return make_absolute(path) if absolute else path


def tmp(absolute: bool = False) -> Path:
    path = Path(".tmp")
    return make_absolute(path) if absolute else path


def test(absolute: bool = False) -> Path:
    return tmp(absolute) / "test"


def make_absolute(path: Union[str, Path]) -> Path:
    path_ = Path(path)

    if path_.is_absolute():
        raise ValueError("Failed to convert relative path to absolute path. The path is already absolute.")

    return project_root() / path


def make_relative(path: Union[str, Path]) -> Path:
    path_ = Path(path)

    if not path_.is_absolute():
        raise ValueError("Failed to convert absolute path to relative path. The path is already relative.")

    return Path(path).relative_to(project_root())


def make_module(path: Union[str, Path]) -> str:
    path_ = Path(path).with_suffix("")

    try:
        relative = path_.relative_to(project_root())
    except ValueError:
        relative = path_

    return relative.as_posix().replace("/", ".")


# patterns
def data_pattern(absolute: bool = False) -> Path:
    path = Path("data") / "*" / "data_"
    return make_absolute(path) if absolute else path


def result_pattern(absolute: bool = False) -> Path:
    path = Path("model") / "*" / "experiment" / "*"
    return make_absolute(path) if absolute else path


# functions on paths
def file_path() -> Path:
    return Path(inspect.stack()[1].filename).resolve()


def dir_path() -> Path:
    return Path(inspect.stack()[1].filename).resolve().parent


@validator_.path_is_dir("dir_")
def has_content(dir_: Union[Path, str]) -> bool:
    return any(Path(dir_).iterdir())


@validator_.path_is_dir("dir_")
def is_empty(dir_: Union[Path, str]) -> bool:
    return not any(Path(dir_).iterdir())


@validator_.path_is_dir("dir_")
def get_content(dir_: Union[Path, str]) -> List[Path]:
    return sorted([item for item in Path(dir_).iterdir()])


@validator_.path_is_dir("dir_")
def get_files(dir_: Union[Path, str]) -> List[Path]:
    d = Path(dir_)

    if not d.is_absolute():
        d = make_absolute(d)

    return sorted([item for item in d.iterdir() if item.is_file()])


@validator_.path_is_dir("dir_")
def get_dirs(dir_: Union[Path, str]) -> List[Path]:
    d = Path(dir_)

    if not d.is_absolute():
        d = make_absolute(d)

    return sorted([item for item in d.iterdir() if item.is_dir()])


@validator_.path_is_dir("dir_")
def get_all_files(
    dir_: Optional[Union[Path, str]] = None,
    excluded_files: Optional[List[Union[str, Path]]] = None,
    excluded_dirs: Optional[List[Union[str, Path]]] = None,
) -> List[Path]:
    if dir_ is None:
        dir_ = project_root()

    d = Path(dir_)

    if excluded_files is None:
        excluded_files = []

    if excluded_dirs is None:
        excluded_dirs = []

    files = [
        file
        for file in get_files(d)
        if not any(make_relative(file).match(str(excluded)) for excluded in excluded_files)
    ]

    dirs = [d_ for d_ in get_dirs(d) if not any(make_relative(d_).match(str(excluded)) for excluded in excluded_dirs)]

    for d in dirs:
        files.extend(get_all_files(dir_=d, excluded_files=excluded_files, excluded_dirs=excluded_dirs))

    return files


@validator_.path_is_dir("dir_")
def get_all_dirs_by_name(dir_: Union[Path, str], name: str) -> List[Path]:
    subdirs = get_dirs(dir_)

    found = [d for d in subdirs if d.name == name]

    for d in subdirs:
        found.extend(get_all_dirs_by_name(dir_=d, name=name))

    return found
