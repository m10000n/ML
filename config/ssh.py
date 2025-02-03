import argparse

_HOST: str = "3.227.240.27"
_PORT: int = 22
_USER: str = "ec2-user"
_IDENTITY_FILE: str = "/Users/mg/.ssh/aws"


class Ssh:
    @staticmethod
    def host() -> str:
        if not _HOST:
            raise ValueError("Host not specified.")
        return _HOST

    @staticmethod
    def port() -> int:
        if not _PORT:
            raise ValueError("Port not specified.")
        return _PORT

    @staticmethod
    def user() -> str:
        if not _USER:
            raise ValueError("User not specified.")
        return _USER

    @staticmethod
    def identity_file() -> str:
        if not _IDENTITY_FILE:
            raise ValueError("Path to identity file not specified.")
        return _IDENTITY_FILE


if __name__ == "__main__":
    commands = {
        "host": Ssh.host,
        "port": Ssh.port,
        "user": Ssh.user,
        "identity_file": Ssh.identity_file,
    }

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=commands.keys(),
    )
    command = parser.parse_args().command

    print(commands[command]())
