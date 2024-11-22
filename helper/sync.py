import argparse
import subprocess
import sys

from config.ssh import HOST, IDENTITY_FILE, PORT, USER
from config.transfer import EXCLUDED_FILES, INCLUDE_DATASET, REMOTE_BASE_DIR
from helper.path import (
    LOCAL_BASE_DIR,
    LOCAL_CONFIG_PATH,
    LOCAL_DATALOADER_PATH,
    LOCAL_DATASET_PATH,
    LOCAL_HELPER_PATH,
    LOCAL_MODEL_PATH,
    LOCAL_TEMP_PATH,
    MODEL_PATH,
    TEMP_PATH,
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

    dependencies = [
        LOCAL_HELPER_PATH,
        LOCAL_CONFIG_PATH,
        *get_file_paths_in_folder(LOCAL_MODEL_PATH),
        *get_file_paths_in_folder(LOCAL_DATALOADER_PATH),
        *get_file_paths_in_folder(LOCAL_TEMP_PATH),
    ]

    if INCLUDE_DATASET:
        dependencies.append(LOCAL_DATASET_PATH)
    else:
        dependencies.extend(get_file_paths_in_folder(LOCAL_DATASET_PATH))

    dependencies = [
        # cast is important otherwise pathlib gets rid of '.'
        f"{LOCAL_BASE_DIR}/./{dependency.relative_to(LOCAL_BASE_DIR)}"
        for dependency in dependencies
        if dependency.is_dir()
        or (dependency.is_file() and dependency.name not in EXCLUDED_FILES)
    ]
    rsync_command.extend(dependencies)

    destination = f"{USER}@{HOST}:{REMOTE_BASE_DIR}"
    rsync_command.append(destination)

    print("----->>Start transfer to training server<<-----")
    transfer(rsync_command)
    print("----->>Finished transfer to training server<<-----")


def from_train():
    rsync_command = [
        "rsync",
        "-av",
        "--progress",
        "--relative",
        "-e",
        f"ssh -p {PORT} -i {IDENTITY_FILE}",
    ]

    dependencies = [MODEL_PATH / "results", TEMP_PATH / "results"]
    dependencies = [
        # cast is important otherwise pathlib gets rid of '.'
        f"{USER}@{HOST}:{REMOTE_BASE_DIR}/./{dependency}"
        for dependency in dependencies
        if dependency.is_dir()
        or (dependency.is_file() and dependency not in EXCLUDED_FILES)
    ]
    rsync_command.extend(dependencies)

    rsync_command.append(LOCAL_BASE_DIR)

    print("----->>Start transfer from training server<<-----")
    transfer(rsync_command=rsync_command)
    print("----->>Finished transfer from training server<<-----")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transfer of files")
    parser.add_argument(
        "command",
        choices=["to_train", "from_train"],
        help="Specify which transfer to make.",
    )
    args = parser.parse_args()

    if args.command == "to_train":
        to_train()
    if args.command == "from_train":
        from_train()
