import click

from helper import history as history_
from helper.clt.custom import CustomCommand


@click.command(cls=CustomCommand, help="Add useful commands to the Bash history.")
def history() -> None:
    history_.__append()
