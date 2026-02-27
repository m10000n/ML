from pathlib import Path

import click

from helper import type_check as type_check_
from helper.clt import validator
from helper.clt.custom import CustomCommand


@click.command(cls=CustomCommand, help="Run static type checking.")
@click.option(
    "--exclude",
    "-e",
    multiple=True,
    type=click.Path(),
    help="Exclude the given path from type checking.",
)
@validator.relative_paths_exist("exclude")
def type_check(exclude: list[str]) -> None:
    exclude_ = [Path(exclude) for exclude in exclude]
    type_check_.__main(exclude=exclude_)
