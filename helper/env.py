import subprocess
import sys


def save_dependencies(path):
    shell_script = "save_dependencies.sh"

    try:
        subprocess.run(
            [shell_script, path],
            check=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e.stderr.decode()}", file=sys.stderr)
