import sys
import time
from pathlib import Path

from config import project as project_config
from helper import aws, lock, process
from helper import time as time_
from helper import tmux
from helper.print import print_error, print_info
from helper.process import PROCESS_LOCK
from helper.tmux import in_tmux

##### config start #####
_SHUTDOWN_CHECK_INTERVAL_SECONDS = 10
_SHUTDOWN_DELAY_SECONDS = 60
##### config end #####

_STOP_FUNCTIONS = {"aws": aws.stop_instance}

_WINDOW_NAME = "stop"

_LOCK = lock.get_lock("stop")
_PROCESS_ID = process.get_reserved_id("stop")
_PROCESS_IDENTIFIER = "STOP"


def stop(dry_run: bool) -> None:
    try:
        provider = project_config.get_provider()
    except ValueError:
        print_error("If no provider is specified, the server cannot be stopped.")
        sys.exit(1)

    if provider not in _STOP_FUNCTIONS:
        print_error(
            f"Failed to stop instance. Your provider `{provider}` is not supported. Feel free to add this functionality in: {Path(__file__)}"
        )
        sys.exit(1)
    else:
        _STOP_FUNCTIONS[provider](dry_run)


def is_active() -> bool:
    return process.is_tracked(_PROCESS_ID)


@in_tmux(
    _WINDOW_NAME,
    process_id=_PROCESS_ID,
    process_command=_PROCESS_IDENTIFIER,
    process_important=False,
    attach_if_exists=True,
)
def __activate() -> None:
    with _LOCK:
        try:
            stop(dry_run=True)
        except RuntimeError as e:
            print_error(f"Failed to activate stop. {e}")
            sys.exit(1)

        print_info(f"Activated stop.")
        print()

    while True:
        print(
            "\033[F"
            + f"[{time_.now_str()}] Number of tracked processes: {process.get_n_tracked_important()}".ljust(80),
            flush=True,
        )
        if process.get_n_tracked_important() == 0:
            print_info(f"This instance will be stopped in {_SHUTDOWN_DELAY_SECONDS} seconds.")
            time.sleep(_SHUTDOWN_DELAY_SECONDS)
            with _LOCK:
                with PROCESS_LOCK:
                    if process.get_n_tracked_important(with_lock=False) == 0:
                        print_info("Stopping instance...")
                        process.get(_PROCESS_ID, with_lock=False).untrack(with_lock=False)
                        tmux.detach_all()
                        stop(dry_run=False)
        time.sleep(_SHUTDOWN_CHECK_INTERVAL_SECONDS)


def __deactivate() -> None:
    with PROCESS_LOCK:
        if process.is_tracked(_PROCESS_ID, with_lock=False):
            process.get(_PROCESS_ID, with_lock=False).kill(with_lock=False)
            tmux.kill_window(_WINDOW_NAME)
            print_info("Deactivated stop.")
        else:
            print_error("Failed to deactivate stop. Stop is not active.")
            sys.exit(1)


@process.not_important
def __info() -> None:
    if process.is_tracked(_PROCESS_ID, with_lock=False):
        print_info(f"Stop is active. Number of tracked processes: {process.get_n_tracked_important()}")
    else:
        print_info(f"Stop is not active.")
