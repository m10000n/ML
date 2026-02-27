import click

from helper import debug as debug_
from helper import ssh as ssh_
from helper.clt.custom import CustomGroup


@click.group(cls=CustomGroup, help="Manage the debug environment.")
def debug() -> None:
    pass


@debug.command(help="Activate the debug environment.")
@click.option("--remote", "-r", is_flag=True, help="Enable remote debugging mode")
def activate(remote: bool) -> None:
    debug_.__activate(remote)


@debug.command(help="Deactivate the debug environment.")
def deactivate() -> None:
    debug_.__deactivate()


@debug.command(help="Show information about the current debug environment status.")
def info() -> None:
    debug_.__info()


# SSH tunnel
@debug.group(cls=CustomGroup, help="Manage the SSH tunnel for remote debugging.")
def tunnel() -> None:
    pass


@tunnel.command(name="info", help="Show information about the SSH tunnel.")
def tunnel_info() -> None:
    ssh_.__tunnel_info()


@tunnel.command(name="start", help="Start the SSH tunnel.")
def tunnel_start() -> None:
    ssh_.__start_tunnel()


@tunnel.command(name="stop", help="Stop the SSH tunnel.")
def tunnel_stop() -> None:
    ssh_.__stop_tunnel()
