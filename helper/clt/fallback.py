import shutil
import sys
import textwrap
from typing import List, NoReturn, Sequence, Tuple, Union

from helper import env, path, setup

_LINE_WIDTH = max(shutil.get_terminal_size(fallback=(80, 24)).columns, 20)

_HELP = """\
Usage: ML COMMAND [ARGS]...

{description}

Options:
{options}"""

_HELP_COMMANDS = """
Commands:
{commands}"""

_HELP_ERROR = """\
Usage: ML COMMAND [ARGS]...
Try 'ML {arguments}--help' for help.

Error: {error}"""

_ENV_CONFIG_PATH_RELATIVE = path.make_relative(env._CONFIG_PATH)


def main(*args: str) -> None:
    description = ""
    options = []
    commands = []

    env_description = "Manage the Python environment."
    setup_description = "Setup the project."

    if not args or args[0] == "--help":
        description = "This command line tool streamlines machine learning workflows and development."
        commands = [("env", env_description), ("setup", setup_description)]
    elif args[0] == "env":
        create_description = f"Create a new Python environment using `{_ENV_CONFIG_PATH_RELATIVE}`."
        update_description = f"Update the Python environment using `{_ENV_CONFIG_PATH_RELATIVE}`."
        if len(args) == 1 or args[1] == "--help":
            description = env_description
            commands = [("create", create_description), ("update", update_description)]
        elif args[1] == "create":
            if len(args) == 2:
                env.__create()
            elif args[2] == "--help":
                description = create_description
            else:
                _print_error_argument(argument=args[2], valid_arguments=args[0:2])
        elif args[1] == "update":
            if len(args) == 2:
                env.__update()
            elif args[2] == "--help":
                description = update_description
            else:
                _print_error_argument(argument=args[2], valid_arguments=args[0:2])
        else:
            _print_error_command(command=args[1], valid_arguments=args[0:1])

    elif args[0] == "setup":
        if "--help" in args:
            description = "Setup the project."
            options = [("install", "Install dependencies.")]
        elif len(args) == 1:
            setup.__main(install=False)
        elif len(args) == 2:
            if args[1] == "--install":
                setup.__main(install=True)
            else:
                _print_error_argument(argument=args[1], valid_arguments=args[0:1])
        else:
            _print_error_argument(argument=args[1], valid_arguments=args[0:1])
    else:
        _print_error_command(command=args[0])

    if description:
        _print_help(description=description, options=options, commands=commands)


def _print_help(
    description: str,
    options: Union[List[Tuple[str, str]], None] = None,
    commands: Union[List[Tuple[str, str]], None] = None,
) -> None:
    if options is None:
        options = []
    if commands is None:
        commands = []

    description_ = _wrap_and_indent(description, 2)

    options.append(("help", "Show this message and exit."))
    options_ = _format_argument([(f"--{option}", description) for option, description in options])

    help = _HELP.format(description=description_, options=options_)

    if commands:
        commands_ = _format_argument(commands)
        help += _HELP_COMMANDS.format(commands=commands_)

    print(help)


def _print_error_command(command: str, valid_arguments: Union[Sequence[str], None] = None) -> None:
    if valid_arguments is None:
        valid_arguments = []
    error_message = f"No such command '{command}'."
    _print_error(error_message=error_message, valid_arguments=list(valid_arguments))


def _print_error_argument(argument: str, valid_arguments: Union[Sequence[str], None] = None) -> None:
    if valid_arguments is None:
        valid_arguments = []
    error_message = f"Got unexpected extra argument ({argument})"
    _print_error(error_message=error_message, valid_arguments=list(valid_arguments))


def _print_error(error_message: str, valid_arguments: List[str]) -> NoReturn:
    valid_arguments_ = " ".join(valid_arguments) + " " if valid_arguments else ""
    print(_HELP_ERROR.format(arguments=valid_arguments_, error=error_message))
    sys.exit(2)


def _wrap_and_indent(text: str, indent: int) -> str:
    wrapped = textwrap.fill(text, width=_LINE_WIDTH - indent)
    return textwrap.indent(wrapped, " " * indent)


def _truncate(line: str, max_length: int) -> str:
    while len(line) > max_length:
        line = textwrap.shorten(line, width=max_length, placeholder="") + "..."
    return line


def _format_argument(arguments: List[Tuple[str, str]]) -> str:
    if not arguments:
        return ""
    longest_argument = max(len(argument) for argument, _ in arguments)
    arguments = [
        (argument, _truncate(description, _LINE_WIDTH - longest_argument - 4)) for argument, description in arguments
    ]
    return "\n".join(
        [f"{' ' * 2}{argument}".ljust(longest_argument + 4) + description for argument, description in arguments]
    )


if __name__ == "__main__":
    main(*sys.argv[1:])
