import argparse
import subprocess
import sys

from config.ssh import HOST, IDENTITY_FILE, PORT, USER
from config.transfer import EXCLUDED_FILES, INCLUDE_DATASET, REMOTE_BASE_DIR
from helper.path import (
    BASE_DIR,
    CONFIG_PATH_A,
    DATALOADER_PATH_A,
    DATASET_PATH_A,
    HELPER_PATH_A,
    MODEL_PATH_A,
    RESULT_PATH_R,
    get_file_paths_in_folder,
)


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
        "--relative",
        "-e",
        f"ssh -p {PORT} -i {IDENTITY_FILE}",
    ]
    rsync_command.extend(
        [f"--exclude={excluded_file}" for excluded_file in EXCLUDED_FILES]
    )

    dependencies = [
        HELPER_PATH_A,
        CONFIG_PATH_A,
        *get_file_paths_in_folder(MODEL_PATH_A),
        *get_file_paths_in_folder(DATALOADER_PATH_A),
    ]

    if INCLUDE_DATASET:
        dependencies.append(DATASET_PATH_A)
    else:
        dependencies.extend(get_file_paths_in_folder(DATASET_PATH_A))

    dependencies = [
        # cast is important otherwise pathlib gets rid of '.'
        f"{BASE_DIR}/./{dependency.relative_to(BASE_DIR)}"
        for dependency in dependencies
    ]
    rsync_command.extend(dependencies)

    destination = f"{USER}@{HOST}:{REMOTE_BASE_DIR}"
    rsync_command.append(destination)

    print("----->>> Start transfer to training server. <<<-----")
    transfer(rsync_command)
    print("----->>> Finished transfer to training server. <<<-----")


def from_train():
    rsync_command = [
        "rsync",
        "-av",
        "--progress",
        "--relative",
        "-e",
        f"ssh -p {PORT} -i {IDENTITY_FILE}",
    ]

    dependencies = [RESULT_PATH_R]
    dependencies = [
        # cast is important otherwise pathlib gets rid of '.'
        f"{USER}@{HOST}:{REMOTE_BASE_DIR}/./{dependency}"
        for dependency in dependencies
    ]

    rsync_command.extend(dependencies)

    rsync_command.append(BASE_DIR)

    print("----->>> Start transfer from training server. <<<-----")
    transfer(rsync_command=rsync_command)
    print("----->>> Finished transfer from training server <<<-----")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transfer of files")
    parser.add_argument("command", choices=["to_train", "from_train"])
    args = parser.parse_args()

    if args.command == "to_train":
        to_train()
    if args.command == "from_train":
        from_train()
