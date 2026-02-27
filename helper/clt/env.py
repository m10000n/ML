import click

from helper import env as env_
from helper import path
from helper.clt.custom import CustomGroup

_ENV_CONFIF_PATH_RELATIVE = path.make_relative(env_._CONFIG_PATH)


@click.group(cls=CustomGroup, help="Manage the Python environment.")
def env() -> None:
    pass


@env.command(help=f"Create a new python environment using `{_ENV_CONFIF_PATH_RELATIVE}`.")
def create() -> None:
    env_.__create()


@env.command(help=f"Update the python environment using `{_ENV_CONFIF_PATH_RELATIVE}`.")
def update() -> None:
    env_.__update()
