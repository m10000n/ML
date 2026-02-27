import click

from helper import count as count_
from helper.clt.custom import CustomCommand


@click.command(cls=CustomCommand, help="Count the lines of code.")
def count() -> None:
    count_.lines()
