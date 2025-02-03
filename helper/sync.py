import argparse
import subprocess
import sys

from config.ssh import Ssh
from config.transfer import Transfer
from helper.path import Path
from helper.print import print_end, print_start


def transfer(rsync_command):
    process = subprocess.Popen(
        rsync_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    for line in process.stdout:
        print(line, end="")
    for line in process.stderr:
        print(line, end="", file=sys.stderr)

    process.wait()

    if process.returncode != 0:
        print(f"rsync failed with return code {process.returncode}", file=sys.stderr)


def to_train():
    rsync_command = [
        "rsync",
        "-av",
        "--progress",
        "-e",
        f"ssh -p {Ssh.port()} -i {Ssh.identity_file()}",
    ]

    excluded_paths = Transfer.excluded_paths()

    if not Transfer.include_dataset():
        excluded_paths.append("data/**/data")

    if not Transfer.include_result():
        excluded_paths.append("model/**/result")

    excluded_paths.extend([".gitignore", ".git"])

    rsync_command.extend(
        [f"--exclude={excluded_path}" for excluded_path in excluded_paths]
    )

    rsync_command.append(f"{Path.project_root()}/")

    rsync_command.append(f"{Ssh.user()}@{Ssh.host()}:{Transfer.remote_project_root()}")

    print_start(text="Start transfer to server.")
    transfer(rsync_command)
    print_end(text="Finished transfer to server.")


def copy_result():
    rsync_command = [
        "rsync",
        "-av",
        "--progress",
        "--relative",
        "-e",
        f"ssh -p {Ssh.port()} -i {Ssh.identity_file()}",
        # cast is important otherwise pathlib gets rid of '.'
        f"{Ssh.user()}@{Ssh.host()}:{Transfer.remote_project_root()}/./{Path.result()}",
        Path.project_root(),
    ]

    print_start(text="Start copying results from server.")
    transfer(rsync_command=rsync_command)
    print_end(text="Finished copying results from server.")


if __name__ == "__main__":
    commands = {"to_train": to_train, "copy_result": copy_result}

    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=commands.keys())
    command = parser.parse_args().command

    commands[command]()
