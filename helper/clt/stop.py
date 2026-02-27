import click

from helper import stop as stop_
from helper.clt.custom import CustomGroup


@click.command(cls=CustomGroup, help="Manage automatic system shutdown based on process activity.")
def stop() -> None:
    pass


@stop.command(help="Show information about the current shutdown configuration.")
def info() -> None:
    stop_.__info()


@stop.command(help="Activate the automatic system shutdown.")
def activate() -> None:
    stop_.__activate()


@stop.command(help="Deactivate the automatic system shutdown.")
def deactivate() -> None:
    stop_.__deactivate()
