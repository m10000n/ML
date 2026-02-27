import sys

from helper import path, shell
from helper.clt.exception import CommandFailed
from helper.print import print_end, print_error, print_start

##### config start #####
_LINE_LENGTH = 120
##### config end #####


def __main() -> None:
    print_start(text="Start cleaning up Python code.", mode="primary")

    success = True

    try:
        print_start("Start removing unused imports.")
        shell.run_command(
            [
                "autoflake",
                "--remove-all-unused-imports",
                "--recursive",
                "--in-place",
                path.project_root(),
            ]
        )
        print_end("Finished removing unused imports.")
    except CommandFailed as e:
        success = False
        print_error(text="Failed to remove unused imports.")
        print(e)

    print()
    try:
        print_start("Start sorting imports.")
        shell.run_command(
            command=[
                "isort",
                path.project_root(),
                "--profile",
                "black",
            ],
            cwd=path.project_root(),
        )
        print_end("Finished sorting imports.")
    except CommandFailed as e:
        success = False
        print_error(text="Failed to sort imports.")
        print(e)

    print()
    try:
        print_start("Start formating.")
        shell.run_command(
            [
                "black",
                path.project_root(),
                "--line-length",
                str(_LINE_LENGTH),
            ]
        )
        print_end("Finished formating.")
    except CommandFailed as e:
        success = False
        print_error(text="Failed to format.")
        print(e)

    message_end = "Finished cleaning up Python code."
    if success:
        print_end(text=message_end, mode="primary")
    else:
        print_error(text=message_end, mode="primary")
        sys.exit(1)
