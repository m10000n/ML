# this file is used before the python environment is ready

import argparse
import os
import sys
from signal import SIGINT, SIGTERM, signal
from types import FrameType
from typing import Union

from helper import file, path, system
from helper.print import print_error, print_info

_DEBUG_DIR_TMP = path.tmp(absolute=True) / "debug"
_IS_ACTIVE_PATH = _DEBUG_DIR_TMP / "active"
_IS_REMOTE_PATH = _DEBUG_DIR_TMP / "remote"


def is_active() -> bool:
    return _IS_ACTIVE_PATH.exists()


def is_remote() -> bool:
    return _IS_REMOTE_PATH.exists()


def enable() -> None:
    if is_active() and is_remote():
        os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"
        start()


def start() -> None:
    import debugpy

    from config import ssh as ssh_config

    if system.port_is_in_use(port=ssh_config.get_port_debug()):
        print_error("Failed to start debugger. Port is already in use.")
        sys.exit(1)

    signal(signalnum=SIGTERM, handler=_shutdown_handler)
    signal(signalnum=SIGINT, handler=_shutdown_handler)

    debugpy.listen(("localhost", ssh_config.get_port_debug()))
    print(f"Debugger listening on port {ssh_config.get_port_debug()}...")
    debugpy.wait_for_client()
    print("Debugger attached. Running script.")


def __activate(remote: bool = False) -> None:
    if is_active() and is_remote() == remote:
        print_error(f"Failed to activate {_get_mode()} debugging. Debugging is already active.")
        sys.exit(1)
    else:
        if not is_active():
            file.touch(_IS_ACTIVE_PATH, exists_ok=False, lock=True)

        if remote and not is_remote():
            file.touch(_IS_REMOTE_PATH, exists_ok=False, lock=True)
        elif not remote and is_remote():
            _IS_REMOTE_PATH.unlink()

        print_info(f"Activated {_get_mode()} debugging.")


def __deactivate() -> None:
    if is_active():
        _IS_ACTIVE_PATH.unlink()
        if is_remote():
            _IS_REMOTE_PATH.unlink()
        print_info("Deactivated debugging.")
    else:
        print_error("Failed to deactivate debugging. Debugging is not active.")
        sys.exit(1)


def __info() -> None:
    if is_active():
        mode = _get_mode().capitalize()
        print_info(f"{mode} debugging is active.")
    else:
        print_info(f"Debugging is not active.")


def _get_mode() -> str:
    return "remote" if is_remote() else "local"


def _shutdown_handler(signum: int, frame: Union[FrameType, None]) -> None:
    print("\rStopping debug session.")
    sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["is_active"])

    args = parser.parse_args()

    if args.command == "is_active":
        sys.exit(0 if is_active() else 1)
