# this file is used before the python environment is ready

import os
from contextlib import nullcontext
from typing import ContextManager

from helper import path


def get_lock(name: str) -> ContextManager:
    try:
        from filelock import FileLock
    except ImportError:
        return nullcontext()

    lock_dir = path.tmp(absolute=True) / "lock"
    os.makedirs(lock_dir, exist_ok=True)
    return FileLock(lock_dir / name)
