import getpass
import os


def get_password(prompt: str) -> str:
    password = ""

    while not password:
        password = getpass.getpass(prompt=prompt + ": ")

        if not password:
            print("Password must not be empty.")

    return password


def get_input(prompt: str) -> str:
    input_ = ""

    while not input_:
        input_ = input(prompt + ": ")

        if not input_:
            print("Input must not be empty.")

    return input_


def get_env_var(key: str) -> str:
    var = os.getenv(key)

    if not var:
        raise ValueError(f"Missing environment variable: '{key}'.")

    return var


def env_vars_exist(keys: list[str]) -> bool:
    missing_keys = [key for key in keys if os.getenv(key) is None]

    if missing_keys:
        raise ValueError(f"Missing environment variable(s): {', '.join(f"`{key}`" for key in missing_keys)}.")

    return True
