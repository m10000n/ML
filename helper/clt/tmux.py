import click

from helper import tmux as tmux_
from helper.clt.custom import CustomGroup


@click.group(cls=CustomGroup, help="Manage the tmux session.")
def tmux() -> None:
    pass


@tmux.command()
@click.argument("window_name", required=False)
def attach(window_name: str | None = None) -> None:
    """
    Attach to the tmux session.

    - Provide a window name to attach to a specific window. If omitted, the first tmux window will be used.
    """
    tmux_.__attach(window_name)


@tmux.command(help="Clear the current tmux pane.")
def clear() -> None:
    tmux_.__clear()


@tmux.command()
@click.argument("target_name")
@click.argument("pane_idx", required=False, type=click.IntRange(min=0))
@click.option(
    "--force",
    "-f",
    is_flag=True,
    help="Required when killing the project's session or a window with multiple panes.",
)
def kill(target_name: str, pane_idx: int | None = 0, force: bool = False) -> None:
    """
    Kill a tmux session, window, or pane.

    - Provide `target_name` as 'SESSION' to kill the session. You must use `--force` in this case.\n
    - Provide `target_name` as a window name to kill a window. Use `--force` to kill a window that has multiple panes.\n
    - Provide `target_name` as a window name and specify a pane index to kill a specific pane.
    """
    if target_name.lower() == "session":
        if not force:
            raise click.UsageError("You must set --force to kill the tmux session.")
        if pane_idx is not None:
            raise click.UsageError("You cannot set a pane index when killing the the tmux session.")

    tmux_.__kill(target_name=target_name, pane=pane_idx, force=force)


@tmux.command(name="list", help="List all windows of the tmux session.")
def list_windows() -> None:
    tmux_.__list_windows()


@tmux.command()
@click.argument("window_name", required=False)
@click.option("--force", "-f", is_flag=True, help="Create a new pane if the window already exists.")
def new(window_name: str | None = None, force: bool = False) -> None:
    """
    Create a new tmux session, window, or pane.

    - Omit `window_name` to create a new session.\n
    - Provide a window name to create a new window. Use `--force` to create a new pane in an existing window.
    """
    if force and not window_name:
        raise click.UsageError("--force is only allowed if window_name is provided.")

    tmux_.__new(window_name=window_name, force=force)


@tmux.command(help="Write a log file for the current tmux pane.")
def log() -> None:
    tmux_.__write_log()
