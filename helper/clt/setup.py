import click

from helper import setup as setup_
from helper.clt.custom import CustomCommand


@click.command(cls=CustomCommand, help="Run initial setup.")
@click.option("--install", "-i", is_flag=True, help="Install dependencies.")
def setup(install: bool) -> None:
    setup_.__main(install=install)
