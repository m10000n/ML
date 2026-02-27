from collections import OrderedDict
from typing import Any, Callable, List, ParamSpec, TypeVar, overload

import click

_P = ParamSpec("_P")
_R = TypeVar("_R")

_USAGE = "Usage: ML COMMAND [ARGS]..."


class CustomCommand(click.Command):
    def format_usage(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        formatter.write_text(_USAGE)


class CustomGroup(click.Group):
    def make_context(
        self,
        info_name: str | None,
        args: list[str],
        parent: click.Context | None = None,
        **extra: Any,
    ) -> click.Context:
        if parent is None:
            info_name = "ML"
        return super().make_context(info_name, args, parent=parent, **extra)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.commands = OrderedDict()

    def format_usage(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        formatter.write_text(_USAGE)

    def list_commands(self, ctx: click.Context) -> List[str]:
        return list(self.commands.keys())

    @overload
    def command(self, f: Callable[_P, _R], /) -> click.Command: ...
    @overload
    def command(self, *args: Any, **kwargs: Any) -> Callable[[Callable[_P, _R]], click.Command]: ...

    def command(self, *args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("cls", CustomCommand)
        return super().command(*args, **kwargs)

    @overload
    def group(self, f: Callable[_P, _R], /) -> click.Group: ...
    @overload
    def group(self, *args: Any, **kwargs: Any) -> Callable[[Callable[_P, _R]], click.Group]: ...

    def group(self, *args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("cls", CustomGroup)
        return super().group(*args, **kwargs)
