import re
import sys
from pathlib import Path

from helper import path, shell
from helper.clt.exception import CommandFailed
from helper.print import print_end, print_error, print_start


def __main(exclude: list[Path] = []) -> None:
    print_start("Start type checking.")

    if exclude:
        print("Excluded paths:")
        print("\n".join([str(excluded) for excluded in exclude]))
        print()

    command: list[str | Path] = [
        "mypy",
        "--explicit-package-bases",
        path.project_root(),
        "--ignore-missing-imports",
        "--disallow-untyped-defs",
    ]

    if exclude:
        exclude_pattern = "|".join(re.escape(p.as_posix().rstrip("/")) + "/?" for p in exclude)
        command.append(f"--exclude={exclude_pattern}")

    end_message = "Finished type checking."
    try:
        shell.run_command_std(command=command, cwd=path.project_root())
        print_end(text=end_message)
    except CommandFailed as e:
        if e.get_exit_code() == 1:
            print_end(text=end_message)
            sys.exit(0)
        else:
            print_error(text="Type checking failed.")
            print(e)
            sys.exit(1)
