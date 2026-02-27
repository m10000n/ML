# this file is used before the python environment is ready

import sys
import threading
import time
from typing import Literal, Optional

_MODES = {
    "primary": "---------->>>>> {text} <<<<<----------",
    "secondary": "----->>> {text} <<<-----",
}

_START = "\033[{format};{color}m"
_RESET = "\033[0m"

_COLORS = {
    "gray": 90,
    "green": 32,
    "magenta": 35,
    "red": 31,
    "yellow": 33,
}


def print_color(text: str, color_: Literal["gray", "green", "magenta", "red", "yellow"], format: int = 0) -> None:
    print(f"{_START.format(format=format, color=_COLORS[color_])}{text}{_RESET}")


def print_start(text: str, mode: Literal["primary", "secondary"] = "secondary") -> None:
    _print(text=text, color="gray", mode=mode)


def print_end(text: str, mode: Literal["primary", "secondary"] = "secondary") -> None:
    _print(text=text, color="green", mode=mode)


def print_info(text: str, mode: Literal["primary", "secondary"] = "secondary") -> None:
    _print(text=text, color="magenta", mode=mode)


def print_error(text: str, mode: Literal["primary", "secondary"] = "secondary") -> None:
    _print(text=text, color="red", mode=mode)


def print_warning(text: str, mode: Literal["primary", "secondary"] = "secondary") -> None:
    _print(text=text, color="yellow", mode=mode)


def _print(
    text: str, color: Literal["gray", "green", "magenta", "red", "yellow"], mode: Literal["primary", "secondary"]
) -> None:
    text = _MODES[mode].format(text=text)
    print_color(text=text, color_=color, format=1)


class Spinner:
    def __init__(self) -> None:
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        def _spin() -> None:
            spinner_chars = ["|", "/", "-", "\\"]
            i = 0
            while not self._stop_event.is_set():
                sys.stdout.write("\r" + spinner_chars[i % len(spinner_chars)])
                sys.stdout.flush()
                time.sleep(0.1)
                i += 1
            sys.stdout.write("\r")
            sys.stdout.flush()

        self._thread = threading.Thread(target=_spin)
        self._thread.start()

    def stop(self) -> None:
        if self._thread:
            self._stop_event.set()
            self._thread.join()
            self._stop_event.clear()
            self._thread = None
            sys.stdout.write("\r \r")
            sys.stdout.flush()
