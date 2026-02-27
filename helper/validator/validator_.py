# this file is used before the python environment is ready

from functools import wraps
from inspect import signature
from pathlib import Path
from typing import Any, Callable, TypeVar

from helper.exception import ValidationError

_R = TypeVar("_R")


def path_exists(param_name: str) -> Callable[[Callable[..., _R]], Callable[..., _R]]:
    def decorator(func: Callable[..., _R]) -> Callable[..., _R]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> _R:
            bound_args = signature(func).bind(*args, **kwargs)
            bound_args.apply_defaults()
            path = bound_args.arguments[param_name]

            if path and not Path(path).exists():
                raise ValidationError(
                    param_name=param_name,
                    constraint=f"The path `{path}` does exist.",
                    value=f"The path does not exist.",
                )

            return func(*args, **kwargs)

        return wrapper

    return decorator


def path_is_dir(param_name: str) -> Callable[[Callable[..., _R]], Callable[..., _R]]:
    def decorator(func: Callable[..., _R]) -> Callable[..., _R]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> _R:
            bound_args = signature(func).bind(*args, **kwargs)
            bound_args.apply_defaults()
            path = bound_args.arguments[param_name]

            if path:
                path = Path(path)
                if not path.exists():
                    raise ValidationError(
                        param_name=param_name,
                        constraint=f"The path `{path}` does exist.",
                        value=f"The path does not exist.",
                    )
                if not path.is_dir():
                    raise ValidationError(
                        param_name=param_name,
                        constraint=f"The path `{path}` is a directory.",
                        value=f"The path is not a directory.",
                    )

            return func(*args, **kwargs)

        return wrapper

    return decorator


def path_is_file(param_name: str) -> Callable[[Callable[..., _R]], Callable[..., _R]]:
    def decorator(func: Callable[..., _R]) -> Callable[..., _R]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> _R:
            bound_args = signature(func).bind(*args, **kwargs)
            bound_args.apply_defaults()
            path = bound_args.arguments[param_name]

            if path:
                path = Path(path)
                if not path.exists():
                    raise ValidationError(
                        param_name=param_name,
                        constraint=f"The path `{path}` does exist.",
                        value=f"The path does not exist.",
                    )
                if not path.is_file():
                    raise ValidationError(
                        param_name=param_name,
                        constraint=f"The path `{path}` is a file.",
                        value=f"The path is not a file.",
                    )

            return func(*args, **kwargs)

        return wrapper

    return decorator
