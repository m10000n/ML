import os
from functools import wraps
from inspect import BoundArguments, signature
from typing import Any, Callable, ParamSpec, TypeVar

import torch

from helper.exception import PreconditionError, ValidationError

_P = ParamSpec("_P")
_R = TypeVar("_R")


# be careful with the constraints, they are evaluated with eval
def constraints(param_name: str, constraints: str | list[str]) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    def decorator(func: Callable[_P, _R]) -> Callable[_P, _R]:
        @wraps(func)
        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            constraints_ = [constraints] if isinstance(constraints, str) else constraints

            bound_args = _get_bound_args(func, *args, **kwargs)
            value = _get_value(bound_args, param_name)

            if value is not None:
                if isinstance(value, (list, tuple)):
                    value_ = value
                elif isinstance(value, torch.Tensor):
                    value_ = value.flatten().tolist()
                else:
                    value_ = [value]

                for v in value_:
                    for constraint in constraints_:
                        if not eval(constraint, {"x": v}):
                            raise ValidationError(
                                param_name=param_name,
                                constraint=constraint,
                                value=f"x = {value}",
                            )

            return func(*args, **kwargs)

        return wrapper

    return decorator


def not_empty(param_name: str) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    def decorator(func: Callable[_P, _R]) -> Callable[_P, _R]:
        @wraps(func)
        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            bound_args = _get_bound_args(func, *args, **kwargs)
            value = _get_value(bound_args, param_name)

            if value is not None:
                if value == "":
                    raise PreconditionError(
                        expected=f"The parameter `{param_name}` is not empty.",
                        actual=f"The parameter is empty.",
                    )

            return func(*args, **kwargs)

        return wrapper

    return decorator


def env_vars_exist(
    env_vars: list[str],
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    def decorator(func: Callable[_P, _R]) -> Callable[_P, _R]:
        @wraps(func)
        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            for env_var in env_vars:
                if not os.getenv(env_var):
                    raise PreconditionError(
                        expected=f"The environment variable `{env_var}` is set.",
                        actual=f"The environmen variable is not set.",
                    )

            return func(*args, **kwargs)

        return wrapper

    return decorator


def is_in(param_name: str, container: list[Any]) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    def decorator(func: Callable[_P, _R]) -> Callable[_P, _R]:
        @wraps(func)
        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            bound_args = _get_bound_args(func, *args, **kwargs)
            value = _get_value(bound_args, param_name)

            if value is not None:
                if value not in container:
                    raise ValidationError(
                        param_name=param_name,
                        constraint=f"The value must be one of {", ".join([f"`{item}`" for item in container])}.",
                        value=f"`{value}`",
                    )

            return func(*args, **kwargs)

        return wrapper

    return decorator


def _get_bound_args(func: Callable[_P, _R], *args: _P.args, **kwargs: _P.kwargs) -> BoundArguments:
    bound_args = signature(func).bind(*args, **kwargs)
    bound_args.apply_defaults()
    return bound_args


def _get_value(bound_args: BoundArguments, param_name: str) -> Any:
    try:
        value = bound_args.arguments[param_name]
    except KeyError as e:
        raise KeyError(f"The parameter `{param_name}` is not defined.") from e
    return value
