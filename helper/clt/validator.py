from functools import wraps
from inspect import signature
from typing import Callable, ParamSpec, TypeVar

import click

from helper import path

_P = ParamSpec("_P")
_R = TypeVar("_R")


def relative_paths_exist(
    param_name: str,
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    def decorator(func: Callable[_P, _R]) -> Callable[_P, _R]:
        @wraps(func)
        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            bound_args = signature(func).bind(*args, **kwargs)
            bound_args.apply_defaults()
            paths = bound_args.arguments.get(param_name, [])

            for path_ in paths:
                if not (path.project_root() / path_).exists():
                    raise click.BadParameter(f"Path '{path_}' does not exist.", param_hint=f"--{param_name}")

            return func(*args, **kwargs)

        return wrapper

    return decorator
