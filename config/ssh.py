import argparse

HOST = "a"
PORT = 1
USER = "a"
IDENTITY_FILE = "a"

if not HOST:
    raise ValueError("You have not specified the host.")

if not PORT:
    raise ValueError("You have not specified the port.")

if not USER:
    raise ValueError("You have not specified the user.")

if not IDENTITY_FILE:
    raise ValueError("You have not specified the path to the identity file.")


if __name__ == "__main__":
    commands = {
        "host": HOST,
        "port": PORT,
        "user": USER,
        "identity_file": IDENTITY_FILE,
    }

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=list(commands.keys()),
    )
    args = parser.parse_args()

    print(commands[args.command])
