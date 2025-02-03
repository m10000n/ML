import argparse
import inspect
from pathlib import Path as PathlibPath

from config.config import Config


class Path:
    # make absolute
    @staticmethod
    def _make_absolute(path: PathlibPath) -> PathlibPath:
        return Path.project_root(absolute=True) / path

    # project root
    @staticmethod
    def project_root(absolute: bool = True) -> PathlibPath:
        return PathlibPath(__file__).parent.parent if absolute else PathlibPath()

    # config
    @staticmethod
    def config(absolute: bool = False) -> PathlibPath:
        path = PathlibPath("config")
        return Path._make_absolute(path) if absolute else path

    # model
    @staticmethod
    def model(absolute: bool = False) -> PathlibPath:
        path = PathlibPath("model") / Config.model()
        return Path._make_absolute(path) if absolute else path

    @staticmethod
    def result(absolute: bool = False) -> PathlibPath:
        path = Path.model() / "result"
        return Path._make_absolute(path) if absolute else path

    # data
    @staticmethod
    def dataset(absolute: bool = False) -> PathlibPath:
        path = PathlibPath("data") / Config.dataset()
        return Path._make_absolute(path) if absolute else path

    @staticmethod
    def data(absolute: bool = False) -> PathlibPath:
        path = Path.dataset() / "data"
        return Path._make_absolute(path) if absolute else path

    # helper
    @staticmethod
    def helper(absolute: bool = False) -> PathlibPath:
        path = PathlibPath("helper")
        return Path._make_absolute(path) if absolute else path

    @staticmethod
    def shell(absolute: bool = False) -> PathlibPath:
        path = Path.helper() / "shell"
        return Path._make_absolute(path) if absolute else path

    @staticmethod
    def env(absolute: bool = False) -> PathlibPath:
        path = Path.helper() / "env"
        return Path._make_absolute(path) if absolute else path

    # path functions
    @staticmethod
    def file_path() -> PathlibPath:
        return PathlibPath(inspect.stack()[1].filename).resolve()

    @staticmethod
    def dir_path() -> PathlibPath:
        return PathlibPath(inspect.stack()[1].filename).resolve().parent

    @staticmethod
    def exists(path: PathlibPath) -> bool:
        return path.exists()

    @staticmethod
    def has_content(path: PathlibPath) -> bool:
        return path.exists() and any(path.iterdir())

    @staticmethod
    def get_content(path: PathlibPath) -> list[PathlibPath]:
        return sorted([item for item in path.iterdir()])

    @staticmethod
    def get_files(path: PathlibPath) -> list[PathlibPath]:
        return sorted([item for item in path.iterdir() if item.is_file()])

    @staticmethod
    def get_folders(path: PathlibPath) -> list[PathlibPath]:
        return sorted([item for item in path.iterdir() if item.is_dir()])


if __name__ == "__main__":
    paths = {
        "project_root": Path.project_root,
        "config": Path.config,
        "model": Path.model,
        "result": Path.result,
        "dataset": Path.dataset,
        "data": Path.data,
        "helper": Path.helper,
        "shell": Path.shell,
        "env": Path.env,
    }

    parser = argparse.ArgumentParser()

    parser.add_argument("path", choices=paths.keys())
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-r", "--relative", action="store_true")
    group.add_argument("-a", "--absolute", action="store_true")
    group.add_argument("-m", "--module", action="store_true")

    args = parser.parse_args()
    path_ = paths[args.path]

    if args.relative:
        print(path_(absolute=False))
    elif args.absolute:
        print(path_(absolute=True))
    elif args.module:
        print(path_(absolute=False).replace("/", "."))
