from pathlib import Path

from config.ssh import Ssh

_INCLUDE_DATASET = False

_INCLUDE_RESULT = False

_EXCLUDED_PATHS = ["**/.DS_Store", "**/__pycache__"]

_REMOTE_PROJECT_ROOT = "/home/{user}/ML"


class Transfer:
    @staticmethod
    def include_dataset() -> bool:
        return _INCLUDE_DATASET

    @staticmethod
    def include_result() -> bool:
        return _INCLUDE_RESULT

    @staticmethod
    def excluded_paths() -> list[Path]:
        return [Path(excluded_path) for excluded_path in _EXCLUDED_PATHS]

    @staticmethod
    def remote_project_root() -> str:
        return _REMOTE_PROJECT_ROOT.format(user=Ssh.user())
