# this file is used before the python environment is ready

import argparse
import sys
from pathlib import Path
from typing import List, cast

from helper import file, path, shell
from helper.clt.exception import CommandFailed
from helper.print import print_end, print_error, print_start

_CONFIG_PATH = path.config(absolute=True) / "env.yml"

_PYTHON_MANAGERS = ["micromamba", "mamba"]
_SHELL_HOOK = 'eval "$({python_manager} shell hook --shell bash)"'


def get_name() -> str:
    confige_file = file.read(_CONFIG_PATH)
    return confige_file.split()[1]


def get_activate_command() -> str:
    manager = _get_python_manager()
    return f"{manager} activate {get_name()}"


def save_dependencies(path: Path, overwrite: bool = False) -> None:
    dependencies = shell.run_command(
        command=[_get_python_manager(), "list", "--export"], verbose=(False, False)
    ).get_stdout(as_list=True)
    dependencies = dependencies[2:]
    file.write_lines(path=path, lines=cast(List[str], dependencies), overwrite=overwrite, lock=True)


def _is_installed() -> bool:
    manager = _get_python_manager()

    envs = shell.run_command([manager, "env", "list"], verbose=(False, False)).get_stdout(as_list=True)

    env_name = get_name()
    for line in envs:
        line_ = line.split()
        if line_ and line_[0] == env_name:
            return True
    return False


def _get_python_manager() -> str:
    for manager in _PYTHON_MANAGERS:
        if shell.is_installed(manager):
            return manager
    raise FileNotFoundError(f"`{_PYTHON_MANAGERS[0]}` is not installed.")


def __is_ready() -> bool:
    try:
        return sys.exit(0) if _is_installed() else sys.exit(1)
    except FileNotFoundError:
        sys.exit(1)


def __setup() -> bool:
    try:
        manager = _get_python_manager()
        shell_hook = _SHELL_HOOK.format(python_manager=manager)

        if _is_installed():
            print_start("Start updating Python environment.")
            command = f"eval \"{shell_hook}\" && {manager} env update --file '{_CONFIG_PATH}' --yes --quiet"
            shell.run_command(command, verbose=(False, False), show_spinner=True)
            print_end("Finished updating Python environment.")
        else:
            print_start("Start creating Python environment.")
            command = f"eval \"{shell_hook}\" && {manager} env create --file '{_CONFIG_PATH}' --yes --quiet"
            shell.run_command(command, verbose=(False, False), show_spinner=True)
            print_end("Finished creating Python environment.")

        return True
    except (FileNotFoundError, CommandFailed) as e:
        print_error("Failed to setup Python environment.")
        print(e)
        return False


def __create() -> None:
    try:
        print_start("Start creating Python environment.")
        if _is_installed():
            raise FileExistsError(f"Python environment `{get_name()}` already exists.")
        else:
            manager = _get_python_manager()
            shell.run_command(
                command=[
                    manager,
                    "env",
                    "create",
                    "--file",
                    _CONFIG_PATH,
                    "--yes",
                    "--quiet",
                ],
                verbose=(False, False),
                show_spinner=True,
            )
            print_end("Finished creating Python environment.")
    except (FileExistsError, FileNotFoundError, CommandFailed) as e:
        print(e)
        print_error("Failed to create Python environment.")
        sys.exit(1)


def __update() -> None:
    try:
        print_start("Start updating Python environment.")
        if not _is_installed():
            raise FileNotFoundError(f"Python environment `{get_name()}` does not exist.")
        else:
            manager = _get_python_manager()
            shell.run_command(
                command=[
                    manager,
                    "env",
                    "update",
                    "--prune",
                    "--file",
                    _CONFIG_PATH,
                    "--yes",
                    "--quiet",
                ],
                verbose=(False, False),
                show_spinner=True,
            )
            print_end("Finished updating Python environment.")
    except (FileNotFoundError, CommandFailed) as e:
        print(e)
        print_error("Failed to update Python environment.")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=[
            "is_ready",
            "get_activate_command",
            "create",
            "update",
        ],
    )

    args = parser.parse_args()

    if args.command == "is_ready":
        __is_ready()
    elif args.command == "get_activate_command":
        print(get_activate_command())
    elif args.command == "create":
        __create()
    elif args.command == "update":
        __update()
