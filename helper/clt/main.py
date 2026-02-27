import sys

import click

from helper import debug as debug_
from helper import system as system_
from helper.clt.count import count
from helper.clt.custom import CustomGroup
from helper.clt.dataset import dataset
from helper.clt.debug import debug
from helper.clt.env import env
from helper.clt.experiment import experiment
from helper.clt.format import format
from helper.clt.history import history
from helper.clt.model import model
from helper.clt.run import run
from helper.clt.setup import setup
from helper.clt.stop import stop
from helper.clt.sync import sync
from helper.clt.system import system
from helper.clt.tmux import tmux
from helper.clt.type_check import type_check
from helper.print import print_info
from helper.process import Process


@click.group(
    cls=CustomGroup,
    help="This command line tool streamlines machine learning workflows and development.",
)
def cli() -> None:
    pass


cli.add_command(count)
cli.add_command(dataset)
cli.add_command(debug)
cli.add_command(env)
cli.add_command(experiment)
cli.add_command(format)
cli.add_command(history)
cli.add_command(model)
cli.add_command(setup)
cli.add_command(stop)
cli.add_command(sync)
cli.add_command(system)
cli.add_command(run)
cli.add_command(tmux)
cli.add_command(type_check)

if __name__ == "__main__":
    system_.init_system()

    if debug_.is_active():
        print_info("Debugging is active.")

    args = sys.argv[1:]
    if not args or not args[0] == "debug":
        debug_.enable()

    process: Process | None = None
    try:
        process = Process(command=f"ML{' ' if args else ''}{' '.join(args)}")
        cli()
    finally:
        if process:
            process.untrack()
