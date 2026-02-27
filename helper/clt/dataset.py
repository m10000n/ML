import click

from helper import data_
from helper.clt.custom import CustomGroup


@click.group(cls=CustomGroup, help="Manage datasets for the current project.")
def dataset() -> None:
    pass


@dataset.command(name="info", help="Show all available datasets.")
@click.option("--pattern", "-p", type=str, default="", help="Show datasets matching the pattern.")
@click.option("--details", "-d", is_flag=True, help="Show detailed information about the datasets.")
def infos(details: bool = False, pattern: str = "") -> None:
    data_.__available_datasets(details=details, pattern=pattern)


@dataset.command()
@click.argument("name")
def size(name: str) -> None:
    """
    Show information about the size of a dataset.

    - `name`: Name of the dataset.
    """
    data_.__size_info(name)


@dataset.command()
@click.argument("name")
def download(name: str) -> None:
    """
    Download a dataset.

    - `name`: Name of the dataset.
    """
    data_.__download(name)


@dataset.command()
@click.argument("name")
def new(name: str) -> None:
    """
    Create a new dataset.

    - `name`: Name of the dataset.
    """
    data_.__new(name)
