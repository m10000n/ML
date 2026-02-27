import click

from helper import format as format_
from helper.clt.custom import CustomCommand


@click.command(
    cls=CustomCommand,
    help="Format Python code, sort imports and remove unused imports.",
)
def format() -> None:
    format_.__main()
