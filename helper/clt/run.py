import click

from helper import path
from helper import run as run_
from helper.clt.custom import CustomCommand


@click.command(
    cls=CustomCommand,
    help=f"Run `main` in `{path.helper(absolute=False) / 'run.py'}`. This can be used for for rapid prototyping.",
)
def run() -> None:
    run_.main()
