# this file is used before the python environment is ready

import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, List, Union

from helper import lock
from helper.validator import validator_

_LOCK = lock.get_lock("file")


# text
@validator_.path_is_file("path")
def read(path: Union[Path, str], unlock: bool = False) -> str:
    with _open_file_with_temp_unlock(path=path, mode="r", unlock=unlock) as file:
        return file.read()


@validator_.path_is_file("path")
def read_lines(path: Union[Path, str], unlock: bool = False) -> List[str]:
    with _open_file_with_temp_unlock(path=path, mode="r", unlock=unlock) as file:
        return [line.strip("\n") for line in file]


def write(path: Union[Path, str], text: str, overwrite: bool = False, lock: bool = False) -> None:
    if text and not text.endswith("\n"):
        text += "\n"

    with _open_file_with_temp_unlock(path=path, mode="w", unlock=overwrite) as file:
        file.write(text)

    if lock:
        set_read(path)


def write_lines(
    path: Union[Path, str],
    lines: List[str],
    overwrite: bool = False,
    lock: bool = False,
) -> None:
    text = "\n".join(lines)
    write(path=path, text=text, overwrite=overwrite, lock=lock)


@validator_.path_is_file("path")
def append(
    path: Union[Path, str],
    text: str,
    check_contains: bool = False,
    unlock: bool = False,
) -> bool:
    if check_contains and text in read(path):
        return False
    else:
        if text and not text.endswith("\n"):
            text += "\n"

        with _open_file_with_temp_unlock(path=path, mode="a", unlock=unlock) as file:
            file.write(text)

        return True


@validator_.path_is_file("path")
def append_lines(
    path: Union[Path, str],
    lines: List[str],
    check_contains: bool = False,
    unlock: bool = False,
) -> bool:

    text = "\n".join(lines)
    return append(path=path, text=text, check_contains=check_contains, unlock=unlock)


# json
@validator_.path_is_file("path")
def read_json(path: Union[Path, str], unlock: bool = False) -> Any:
    with _open_file_with_temp_unlock(path=path, mode="r", unlock=unlock) as file:
        return json.load(fp=file)


def write_json(path: Union[Path, str], data: Any, overwrite: bool = False, lock: bool = False) -> None:
    with _open_file_with_temp_unlock(path=path, mode="w", unlock=overwrite) as file:
        json.dump(obj=data, fp=file, indent=4)

    if lock:
        set_read(path)


# create
def touch(path: Union[Path, str], exists_ok: bool, lock: bool = False) -> None:
    path_ = Path(path)
    dir = path_.parent

    os.makedirs(dir, exist_ok=True)

    if path_.exists() and not exists_ok:
        raise FileExistsError(f"Failed to create new file. The specified path `{path}` already exists.")
    elif not os.access(dir, os.W_OK):
        raise PermissionError(f"Failed to create new file. The specified directory `{dir}` is not writeable.")

    path_.touch(exist_ok=exists_ok)

    if lock:
        set_read(path_)


# permissions
def set_read_write(path: Union[Path, str]) -> None:
    if Path(path).exists():
        with _LOCK:
            os.chmod(path, 0o644)


def set_read(path: Union[Path, str]) -> None:
    if Path(path).exists():
        with _LOCK:
            os.chmod(path, 0o444)


# helper
@contextmanager
def _open_file_with_temp_unlock(path: Union[str, Path], mode: str, unlock: bool = False) -> Iterator:
    with _LOCK:
        path_ = Path(path)

        if not path_.exists():
            if mode == "r":
                raise FileNotFoundError(f"Failed to read file. The specified path `{path}` does not exist.")
            else:
                dir = path_.parent
                os.makedirs(dir, exist_ok=True)
                if not os.access(dir, os.W_OK):
                    raise PermissionError(
                        f"Failed to create new file. The specified directory `{path_.parent}` is not writeable."
                    )
                else:
                    touch(path=path_, exists_ok=False)

        if unlock:
            permissions = os.stat(path_).st_mode & 0o777
            set_read_write(path_)
        else:
            permissions = None

        if mode == "r" and not os.access(path_, os.R_OK):
            raise PermissionError(f"Failed to read file. The specified path `{path}` is not readable.")
        elif mode in ["w", "a"] and not os.access(path_, os.W_OK):
            if mode == "w":
                raise PermissionError(f"Failed to overwrite file. The specified path `{path}` is not writeable.")
            else:
                raise PermissionError(f"Failed to append to file. The specified path `{path}` is not writeable.")

        try:
            with open(path_, mode) as file:
                yield file
        finally:
            if permissions:
                os.chmod(path_, permissions)
