import re

import click

from helper import model_
from helper.clt.custom import CustomGroup


@click.group(cls=CustomGroup, help="Manage experiments for the current project.")
def experiment() -> None:
    pass


# info
@experiment.command(name="info", help="Show all available experiments.")
@click.option("--pattern", "-p", type=str, default="", help="Show experiments matching the pattern.")
@click.option("--details", "-d", is_flag=True, help="Show detailed information about the experiments.")
def experiment_infos(details: bool = False, pattern: str = "") -> None:
    model_.__available_experiments(details, pattern)


# result
@experiment.command(name="result", help="Show results of all finished experiments.")
@click.option("--pattern", "-p", type=str, default="", help="Show experiments matching the regex pattern.")
def experiment_results(pattern: str = "") -> None:
    model_.__experiment_results(pattern)


# run
@experiment.command("run")
@click.argument("name")
@click.option("--cross-validation", "-cv", is_flag=True, help="Use cross-validation.")
@click.option(
    "--track-time",
    "-tt",
    is_flag=True,
    help=(
        "Measure the duration of data loading, forward pass, and backward pass."
        "Note: This option introduces synchronization overhead on GPU, which may slow down the experiment."
    ),
)
def run(name: str, cross_validation: bool, track_time: bool) -> None:
    """
    Run an experiment.

    - `name`: Name of the experiment.
    """
    model_.__run_experiment(exp_name=name, cross_validation=cross_validation, track_time=track_time)


# continue
@experiment.command("continue")
@click.argument("name")
def continue_cv(name: str) -> None:
    """
    Continue cross validation. Unfinished runs will be deleted and restarted.

    - `name`: Name of the experiment including the iteration.
    """
    model_.__continue_cross_validation(name)


# early stop
@experiment.group(cls=CustomGroup, help="Manage early stopping of experiments.")
def early_stop() -> None:
    pass


@early_stop.command("info", help="Show information about early stopping of experiments.")
def early_stop_info() -> None:
    model_.__early_stop_info()


@early_stop.command("activate")
@click.argument("name")
def activate(name: str) -> None:
    """
    Activate early stopping for a running experiment.

    - `name`: Name of the experiment.
    """
    model_.__activate_early_stop(name)


@early_stop.command("deactivate")
@click.argument("name")
def deactivate(name: str) -> None:
    """
    Deactivate early stopping for an experiment.

    - `name`: Name of the experiment.
    """
    model_.__deactivate_early_stop(name)


def _validate_pattern(pattern: str) -> None:
    try:
        re.compile(pattern)
    except re.error as e:
        raise click.BadParameter(f"Invalid regex for --pattern: {e}") from e
