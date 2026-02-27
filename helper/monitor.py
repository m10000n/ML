import os
import sys

import helper.shell as shell
from helper import process, shell, system, tmux
from helper.print import print_error
from helper.tmux import in_tmux


@in_tmux(window_name="htop", attach_if_exists=True, write_log=False)
@process.not_important
def __htop() -> None:
    os.system("htop")


@in_tmux(window_name="sensors", attach_if_exists=True, write_log=False, attach=False)
@process.not_important
def __sensors() -> None:
    try:
        shell.run_command(["watch", "-n", "1", "sensors"])
    except KeyboardInterrupt:
        pass


def __cpu() -> None:
    if system.get_system() == "linux" and (not tmux.session_exists() or not tmux.window_exists("sensors")):
        if not shell.is_installed("sensors"):
            print_error("Failed to display CPU. `sensors` is not installed.")
            sys.exit(1)
        elif not shell.exit_0("sensors"):
            print_error(
                "Failed to display CPU. Failed to dectect sensors. Run `sudo sensors-detect` to read the hardware sensor data."
            )
            sys.exit(1)
        else:
            __sensors()

    if not tmux.window_exists("htop"):
        __htop()
    else:
        tmux.attach("htop")


@in_tmux(window_name="nvidia_smi", attach_if_exists=True, write_log=False)
@process.not_important
def __gpu() -> None:
    if not shell.is_installed("nvidia-smi"):
        print_error("Failed to display GPU usage. `nvidia-smi` is not installed.")
        sys.exit(1)
    else:
        os.system("watch -n 1 nvidia-smi")
