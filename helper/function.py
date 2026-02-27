import importlib
import inspect
import re
from pathlib import Path
from re import Pattern
from types import FunctionType
from typing import List, Tuple

from helper import path


def _get_f(file_path: str | Path, pattern: str | Pattern = "") -> List[Tuple[str, FunctionType]]:
    module = importlib.import_module(path.make_module(file_path))
    funcs = inspect.getmembers(module, inspect.isfunction)
    funcs = [
        (name, func)
        for name, func in funcs
        if inspect.getmodule(func) is module and not name.startswith("_") and re.search(pattern=pattern, string=name)
    ]
    funcs.sort(key=lambda pair: pair[1].__code__.co_firstlineno)
    return funcs
