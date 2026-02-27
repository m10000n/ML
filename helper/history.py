from config import project
from helper import bash, file
from helper.print import print_info


def _get_history() -> list[str]:
    history = []

    try:
        dataset = project.get_dataset()
    except ValueError:
        dataset = None

    if dataset == "HCP_1200":
        history.extend(["export AWS_SECRET_ACCESS_KEY=", "export AWS_ACCESS_KEY_ID="])

    return history


def __append() -> None:
    if bash.HISTORY_PATH.exists():
        file.append_lines(path=bash.HISTORY_PATH, lines=_get_history())
    else:
        file.write_lines(path=bash.HISTORY_PATH, lines=_get_history())

    print_info("Updated bash history.")


def remove_last(n: int) -> None:
    if bash.HISTORY_PATH.exists():
        lines = file.read_lines(path=bash.HISTORY_PATH)
    else:
        lines = []

    lines = lines[:-n]
    file.write_lines(path=bash.HISTORY_PATH, lines=lines)
