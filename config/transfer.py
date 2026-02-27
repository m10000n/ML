from pathlib import Path

from config import ssh

##### config start #####
_REMOTE_PROJECT_ROOT = "/home/{user}/ML"

_INCLUDE_DATASET = False
_INCLUDE_RESULT = False

# relative to project root
_EXCLUDED_FILES: list[str | Path] = ["**.DS_Store"]
_EXCLUDED_DIRS: list[str | Path] = [".vscode"]
##### config end #####


def get_remote_project_root() -> str:
    return _REMOTE_PROJECT_ROOT.format(user=ssh.get_user())


def get_excluded_files() -> list[str | Path]:
    return _EXCLUDED_FILES


def get_excluded_dirs() -> list[str | Path]:
    from helper import path

    excluded = _EXCLUDED_DIRS.copy()

    if not _INCLUDE_DATASET:
        excluded.append(path.data_pattern())

    if not _INCLUDE_RESULT:
        excluded.append(path.result_pattern())

    return excluded
