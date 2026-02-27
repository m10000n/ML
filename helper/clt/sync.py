import click

from helper import transfer as sync_
from helper.clt.custom import CustomGroup


@click.group(
    cls=CustomGroup,
    help="Manage file synchronization between local and remote systems.",
)
def sync() -> None:
    pass


@sync.command(help="Retrieve results from the server.")
def copy_result() -> None:
    sync_.__copy_result()


@sync.command(help="Sync the project to the server.")
def to_server() -> None:
    sync_.__to_server()
