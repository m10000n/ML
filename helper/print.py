from argparse import ArgumentParser

from colorama import Fore, Style

MODES = {
    "primary": "---------->>>>> {text} <<<<<----------",
    "secondary": "----->>> {text} <<<-----",
}
COLORS = {
    "start": Fore.LIGHTBLACK_EX,
    "end": Fore.GREEN,
    "info": Fore.MAGENTA,
    "error": Fore.RED
}

def print_start(text: str, mode: str = "secondary") -> None:
    print_(text=text, color=COLORS["start"], mode=mode)

def print_end(text: str, mode: str = "secondary") -> None:
    print_(text=text, color=COLORS["end"], mode=mode)

def print_info(text: str, mode: str = "secondary") -> None:
    print_(text=text, color=COLORS["info"], mode=mode)

def print_error(text: str, mode: str = "secondary") -> None:
    print_(text=text, color=COLORS["error"], mode=mode)

def print_(text: str, color: Fore, mode: str = "secondary") -> None:
    if mode not in MODES.keys():
        raise ValuesError(f"Invalid mode: '{mode}. Allowed values: {MODES.keys()}")

    print(Style.BRIGHT + color + (MODES[mode]).format(text=text) + Style.RESET_ALL)


if __name__ == "__main__":
    commands = {"start": print_start, "end": print_end, "info": print_info, "error": print_error}

    parser = ArgumentParser()
    parser.add_argument("command", choices=commands.keys())
    parser.add_argument("mode", choices=["primary", "secondary"])
    parser.add_argument("text", type=str)
    args = parser.parse_args()

    commands[args.command](text=args.text, mode=args.mode)
