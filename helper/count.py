from helper import file, path, shell
from helper.clt.exception import CommandFailed
from helper.print import print_error

##### config start #####
# relative to project root
INCLUDED_DIRS: list[str] = ["config", "data", "helper", "model"]
EXCLUDED_FILES: list[str] = ["**.DS_Store"]
EXCLUDED_DIRS: list[str] = []
##### config end #####

_EXCLUDED_FILES = ["**.conf", "**.yml", "**__init__.py"]
_EXCLUDED_DIRS = ["**__pycache__", path.data_pattern(), path.result_pattern()]

_COUNT_PATH = path.tmp(absolute=True) / "count" / "paths.txt"


def lines() -> None:
    included_dirs = [path.make_absolute(dir_) for dir_ in INCLUDED_DIRS]

    all_files = []

    for dir_ in included_dirs:
        all_files.extend(
            [
                str(file)
                for file in path.get_all_files(
                    dir_=dir_,
                    excluded_files=EXCLUDED_FILES + _EXCLUDED_FILES,
                    excluded_dirs=EXCLUDED_DIRS + _EXCLUDED_DIRS,
                )
            ]
        )

    file.write_lines(path=_COUNT_PATH, lines=all_files, overwrite=True, lock=True)

    try:
        shell.run_command_std(["cloc", "--quiet", "--list-file", _COUNT_PATH])
    except CommandFailed as e:
        print_error("Failed to count lines of code.")
        print(e)

    file.write_lines(
        path=_COUNT_PATH, lines=[str(path.make_relative(file)) for file in all_files], overwrite=True, lock=True
    )

    print(f"\nAll counted files can be found in `{path.make_relative(_COUNT_PATH)}`.")
