import subprocess
import sys

from helper.path import SHELL_PATH


def save_dependencies(path):
    shell_script = SHELL_PATH / "save_dependencies.sh"

    try:
        subprocess.run(
            [shell_script, path],
            check=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e.stderr.decode()}", file=sys.stderr)
