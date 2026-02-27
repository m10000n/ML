import sys
from pathlib import Path

from config import ssh as ssh_config
from config import transfer as transfer_config
from helper import file, path, shell, ssh
from helper.clt.exception import CommandFailed
from helper.print import print_end, print_error, print_start

_EXCLUDED_FILES: list[str | Path] = []
_EXCLUDED_DIRS: list[str | Path] = [".temp", ".git", "**.mypy_cache", "**__pycache__"]

_FILES_TO_TRANSFER_PATH = path.tmp(absolute=True) / "files_to_transfer.txt"


def __to_server() -> None:
    dir_command: list[str | Path] = [
        "ssh",
        "-p",
        str(ssh_config.get_port()),
        "-i",
        str(ssh_config.get_identity_file()),
        f"{ssh_config.get_user()}@{ssh_config.get_host_name()}",
        f"mkdir -p {transfer_config.get_remote_project_root()}",
    ]

    excluded_files = transfer_config.get_excluded_files() + _EXCLUDED_FILES
    excluded_dirs = transfer_config.get_excluded_dirs() + _EXCLUDED_DIRS

    files = [
        str(path.make_relative(file))
        for file in path.get_all_files(excluded_files=excluded_files, excluded_dirs=excluded_dirs)
    ]

    rsync_command: list[str | Path] = [
        "rsync",
        "-az",
        "--out-format=%n (%l bytes)",
        "--files-from",
        _FILES_TO_TRANSFER_PATH,
        "-e",
        f"ssh -p {ssh_config.get_port()} -i {ssh_config.get_identity_file()}",
        f"{path.project_root()}/",
        f"{ssh_config.get_user()}@{ssh_config.get_host_name()}:{transfer_config.get_remote_project_root()}",
    ]

    ssh.__setup()
    print()

    try:
        file.write_lines(path=_FILES_TO_TRANSFER_PATH, lines=files)

        print_start("Start transfer to server.")
        shell.run_command(dir_command)
        shell.run_command(rsync_command)
        print_end(text="Finished transfer to server.")
    except CommandFailed as e:
        print_error("Failed to transfer to server.")
        print(e)
        sys.exit(1)
    finally:
        if _FILES_TO_TRANSFER_PATH.exists():
            _FILES_TO_TRANSFER_PATH.unlink()


def __copy_result() -> None:
    command: list[str | Path] = [
        "rsync",
        "-rz",
        "--relative",
        "--out-format=%n (%l bytes)",
        "-e",
        f"ssh -p {ssh_config.get_port()} -i {ssh_config.get_identity_file()}",
        f"{ssh_config.get_user()}@{ssh_config.get_host_name()}:{transfer_config.get_remote_project_root()}/./{path.result_pattern()}/",
        str(path.project_root()),
    ]

    try:
        ssh.__setup()
        print()

        print_start(text="Start copying results from server.", mode="primary")
        shell.run_command(command)
        print_end(text="Finished copying results from server.", mode="primary")
    except CommandFailed as e:
        print(e)
        print_error("Failed to copy results from server.")
        sys.exit(1)
