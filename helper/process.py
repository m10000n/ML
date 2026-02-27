from __future__ import annotations

import uuid
from contextlib import nullcontext
from functools import wraps
from typing import Callable, ParamSpec, TypeVar

import psutil

from helper import file, lock, path

PROCESS_LOCK = lock.get_lock("process")

_P = ParamSpec("_P")
_R = TypeVar("_R")

_TRACKED_PATH = path.tmp(absolute=True) / "process" / "tracked"

_RESERVED_IDS = {
    "stop": "4d22a634-de4c-4e59-9c5b-5beeb672d218",
    "ssh": "440ebd13-f38d-482e-992a-1ceb121911b6",
    "2": "e70e0cc7-41f7-47dc-9140-e640e3072df1",
    "3": "85e02032-691e-4a2c-a7b5-397a36b433c0",
    "4": "fd703c44-0f8c-4582-8717-1a9c2c143516",
    "5": "e7cfa8fd-3a4b-47ae-82cb-46e710b0958b",
    "6": "88316d7b-1467-41ae-a6a9-8f115e33d544",
    "7": "354bc0fd-fd71-47dd-a0d3-28e8eaf70824",
    "8": "aa71d4ad-4bb7-4548-a6d0-f1a2b707ab1b",
    "9": "a139b3eb-71f0-4a86-aab3-7cc1ed3e0638",
}


class Process:
    pid: int
    id_: str
    command: str | None
    important: bool
    _meta: list[str] | None

    def __init__(
        self,
        pid: int | None = None,
        id_: str | None = None,
        command: str | None = None,
        meta: list[str] | str | None = None,
        important: bool = True,
    ) -> None:
        if id_ is not None and "|" in id_:
            raise ValueError("`id_` must not contain '|'.")

        if command is not None and "|" in command:
            raise ValueError("Command must not contain '|'.")

        with PROCESS_LOCK:
            pid = _get_this_pid() if pid is None else pid
            id__ = get_new_id(with_lock=False) if id_ is None else id_

            all_processes = get_all(with_lock=False)
            for tracked in all_processes:
                if pid == tracked.pid:
                    raise RuntimeError(f"Process with pid `{pid}` already tracked.")

                if id__ == tracked.id_:
                    raise RuntimeError(f"Process with id `{id__}` already tracked.")

            self.pid = pid if pid is not None else _get_this_pid()
            self.id_ = get_new_id(with_lock=False) if id_ is None else id_
            self.command = command
            self.meta = [meta] if isinstance(meta, str) else meta
            self.important = important

            all_processes.append(self)
            _write_processes(all_processes)

    def __str__(self) -> str:
        command_ = "" if self.command is None else self.command
        important_ = "1" if self.important else "0"
        meta_ = "" if self.meta is None else ",".join(self.meta)
        return f"{self.pid} | {self.id_} | {command_} | {important_} | {meta_}"

    @staticmethod
    def _from_str(str_: str) -> Process:
        pid, id_, command, important, meta = str_.split(" | ")

        process_info = object.__new__(Process)
        process_info.pid = int(pid)
        process_info.id_ = id_
        process_info.command = None if command == "" else command
        process_info.important = important == "1"
        process_info._meta = None if meta == "" else meta.split(",")

        return process_info

    @property
    def meta(self) -> list[str] | None:
        return self._meta

    @meta.setter
    def meta(self, meta: list[str] | str | None) -> None:
        meta_ = [meta] if isinstance(meta, str) else meta
        self._validate_meta(meta_)
        self._meta = meta_

    def add_meta(self, meta: str | list[str]) -> None:
        old_meta = [] if self.meta is None else self.meta
        self.meta = old_meta + ([meta] if isinstance(meta, str) else meta)

    def is_running(self) -> bool:
        return psutil.pid_exists(self.pid)

    def is_tracked(self, with_lock: bool = True) -> bool:
        return is_tracked(self.id_, with_lock=with_lock)

    def untrack(self, with_lock: bool = True) -> None:
        with PROCESS_LOCK if with_lock else nullcontext():
            if not self.is_tracked(with_lock=False):
                raise RuntimeError(f"`{str(self)}` not tracked.")

            new_processes = [process for process in get_all(with_lock=False) if process.id_ != self.id_]
            _write_processes(processes=new_processes)

    def kill(self, with_lock: bool = True) -> None:
        self.untrack(with_lock=with_lock)
        try:
            psutil.Process(self.pid).kill()
        except psutil.NoSuchProcess:
            pass

    def update(self, with_lock: bool = True) -> None:
        with PROCESS_LOCK if with_lock else nullcontext():
            if not self.is_tracked(with_lock=False):
                raise RuntimeError(f"This process ({self.id_}) is not tracked.")

            processes = [process for process in get_all(with_lock=False) if process.id_ != self.id_]
            processes.append(self)
            _write_processes(processes)

    def _validate_meta(self, meta: list[str] | None) -> None:
        if meta is None:
            return

        if any("|" in m for m in meta):
            raise ValueError("Meta must not contain '|'.")

        if any("," in m for m in meta):
            raise ValueError("Meta must not contain ','.")


def get_reserved_id(key: str) -> str:
    return _RESERVED_IDS[key]


def get_new_id(with_lock: bool = True) -> str:
    with PROCESS_LOCK if with_lock else nullcontext():
        while True:
            new_id = str(uuid.uuid4())
            if not new_id in _RESERVED_IDS.values() and not is_tracked(id_=new_id, with_lock=False):
                return new_id


def get_all(only_important: bool = False, with_lock: bool = True) -> list[Process]:
    with PROCESS_LOCK if with_lock else nullcontext():
        if _TRACKED_PATH.exists():
            all_tracked = [Process._from_str(process) for process in file.read_lines(_TRACKED_PATH, unlock=True)]
        else:
            all_tracked = []

        running = [process for process in all_tracked if process.is_running()]

        if len(running) != len(all_tracked):
            _write_processes(running)

        return [process for process in running if not only_important or process.important]


def get_this(with_lock: bool = True) -> Process:
    with PROCESS_LOCK if with_lock else nullcontext():
        pid = _get_this_pid()
        process = [process for process in get_all(with_lock=False) if process.pid == pid]
        if len(process) == 0:
            raise AssertionError(f"Process with pid `{pid}` not found. This should never happen.")
        elif len(process) > 1:
            raise AssertionError(f"Multiple processes with pid `{pid}` found. This should never happen.")
        return process[0]


def get(id_: str, with_lock: bool = True) -> Process:
    try:
        return next(process for process in get_all(with_lock=with_lock) if process.id_ == id_)
    except StopIteration:
        raise KeyError(f"Process with id `{id_}` not found.")


def get_n_tracked(with_lock: bool = True) -> int:
    return len(get_all(with_lock=with_lock))


def get_n_tracked_important(with_lock: bool = True) -> int:
    return len(get_all(only_important=True, with_lock=with_lock))


def is_tracked(id_: str, with_lock: bool = True) -> bool:
    return any(process.id_ == id_ for process in get_all(with_lock=with_lock))


def not_important(func: Callable[_P, _R]) -> Callable[_P, _R]:
    @wraps(func)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        with PROCESS_LOCK:
            process = get_this(with_lock=False)
            process.important = False
            process.update(with_lock=False)

        return func(*args, **kwargs)

    return wrapper


def _get_this_pid() -> int:
    return psutil.Process().pid


def _write_processes(processes: list[Process]) -> None:
    file.write_lines(path=_TRACKED_PATH, lines=[str(process) for process in processes], overwrite=True, lock=True)
