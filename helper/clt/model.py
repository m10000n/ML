import click

from helper import model_
from helper.clt.custom import CustomGroup


@click.group(cls=CustomGroup, help="Manage machine learning models for the current project.")
def model() -> None:
    pass


@model.command(name="info", help="Show all available models.")
@click.option("--pattern", "-p", type=str, default="", help="Show models matching the pattern.")
@click.option("--details", "-d", is_flag=True, help="Show detailed information about the models.")
def infos(details: bool = False, pattern: str = "") -> None:
    model_.__available_models(details, pattern)


@model.command("summary")
@click.argument("name")
@click.option("--batch-size", "-bs", type=int, default=None, help="Batch size (must be > 0).")
def summary(name: str, batch_size: int) -> None:
    """
    Show a summary of a model.

    - `name`: Name of the model.
    """
    if batch_size is not None and batch_size <= 0:
        raise click.UsageError("Batch size must be > 0.")

    model_.__summary(model_name=name, batch_size=batch_size)


@model.command(name="flops")
@click.argument("name")
@click.option("--batch-size", "-bs", type=int, default=None, help="Batch size (must be > 0). Defaults to 1.")
def flops(name: str, batch_size: int) -> None:
    """
    Show a FLOPs report for a model's forward pass.

    - `name`: Name of the model.
    """

    if batch_size is not None and batch_size <= 0:
        raise click.UsageError("Batch size must be > 0. Defaults to 1.")

    model_.__flops(model_name=name, batch_size=batch_size)


@model.command(name="new", help="Create a new model.")
@click.argument("name")
def new(name: str) -> None:
    model_.__new(name)
