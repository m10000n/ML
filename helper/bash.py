# this file is used before the python environment is ready

import argparse
from typing import List

from helper import file, path, system

BINARIES = "/bin/bash"
CONFIG_PATH = path.home() / ".bashrc" if system.get_system() == "linux" else path.home() / ".bash_profile"
HISTORY_PATH = path.home() / ".bash_history"

_ML = [
    "ML() {",
    '    if [[ -n "$ML_DEFINITION" ]]; then',
    '        eval "$ML_DEFINITION"',
    '        ML_DEFINITION "$@"',
    "    else",
    "        echo 'Please change directory to the ML project.'",
    "    fi",
    "}",
]


def add(text: str) -> bool:
    text = "\n" + text
    if CONFIG_PATH.exists():
        return file.append(path=CONFIG_PATH, text=text, check_contains=True)
    else:
        file.write(path=CONFIG_PATH, text=text)
        return True


def add_lines(lines: List[str]) -> bool:
    lines = [""] + lines
    if CONFIG_PATH.exists():
        return file.append_lines(path=CONFIG_PATH, lines=lines, check_contains=True)
    else:
        file.write_lines(path=CONFIG_PATH, lines=lines)
        return True


def add_ML() -> bool:
    return add_lines(_ML)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=["get_config_path"],
    )
    args = parser.parse_args()

    if args.command == "get_config_path":
        print(CONFIG_PATH)
