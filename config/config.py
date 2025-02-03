import argparse

_MODEL = "Wang"

_DATASET = ""


class Config:
    @staticmethod
    def model() -> str:
        if not _MODEL:
            raise ValueError("Model not specified.")
        return _MODEL

    @staticmethod
    def dataset() -> str:
        if not _DATASET:
            raise ValueError("Dataset not specified.")
        return _DATASET


if __name__ == "__main__":
    commands = {
        "model": Config.model,
        "dataset": Config.dataset,
    }

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=commands.keys(),
    )
    command = parser.parse_args().command

    print(commands[command]())
