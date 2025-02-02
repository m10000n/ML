from colorama import Fore, Style

PRIMARY = "---------->>>>> {text} <<<<<----------"
SECONDARY = "---->>> {text} <<<----"
START_COLOR = Fore.LIGHTBLACK_EX
END_COLOR = Fore.GREEN

def print_start(text: str, primary: bool = False) -> None:
    print(Style.BRIGHT + START_COLOR +  (PRIMARY if primary else SECONDARY).format(text=text) + Style.RESET_ALL)

def print_end(text: str, primary: bool = False) -> None:
    print(Style.BRIGHT + END_COLOR +  (PRIMARY if primary else SECONDARY).format(text=text) + Style.RESET_ALL)
