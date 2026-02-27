import os
import shutil
from pathlib import Path

from helper import file, path

TEMP_PATH = f"{path.tmp(absolute=True)}/{{path.stem}}_temp{{path.suffix}}"


def create_temp(path_: Path) -> None:
    os.makedirs(path.tmp(absolute=True), exist_ok=True)
    shutil.copy(src=path_, dst=TEMP_PATH.format(path=path))


def replace(path_: Path, replacements: dict[int, str]) -> None:
    original_file = file.read_lines(path_)
    new_file = [f"{replacements[i]}\n" if i in replacements else line for i, line in enumerate(original_file)]
    file.write(path=path_, text="".join(new_file))


def restore(path_: Path) -> None:
    temp_path = Path(TEMP_PATH.format(path=path_))
    if not temp_path.exists():
        raise FileNotFoundError(f"Could not restore `{path}` because the backup file does not exist.")

    shutil.copy(src=TEMP_PATH.format(path=path_), dst=path_)
    temp_path.unlink()
    path.tmp(absolute=True).rmdir()
