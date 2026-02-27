# this file is used before the python environment is ready

from pathlib import Path
from typing import List, Union, cast

from helper import shell
from helper.clt.exception import (
    CommandFailed,
    PackageInstallationFailed,
    PackageUpgradeFailed,
)

_PACKAGE_MANAGERS = {
    "apt-get": {
        "is_installed": ["dpkg", "-s"],
        "update": ["sudo", "apt-get", "update", "--quiet"],
        "upgrade": ["sudo", "apt-get", "upgrade", "--yes", "--quiet"],
        "install": ["sudo", "apt-get", "install", "--yes", "--quiet"],
        "clean": ["sudo", "apt-get", "clean", "--quiet"],
    },
    "brew": {
        "is_installed": ["brew", "list", "--versions", "--quiet"],
        "update": ["brew", "update", "--quiet"],
        "upgrade": ["brew", "upgrade", "--quiet"],
        "install": ["brew", "install", "--quiet"],
        "clean": ["brew", "cleanup", "--quiet"],
    },
    "dnf": {
        "is_installed": ["rpm", "-q"],
        "update": ["sudo", "dnf", "makecache", "--quiet"],
        "upgrade": ["sudo", "dnf", "upgrade", "--quiet"],
        "install": ["sudo", "dnf", "install", "-y", "--quiet"],
        "clean": ["sudo", "dnf", "clean", "all", "--quiet"],
    },
    "pacman": {
        "is_installed": ["pacman", "-Qi"],
        "update": ["sudo", "pacman", "-Sy", "--noconfirm", "--quiet"],
        "upgrade": ["sudo", "pacman", "-Su", "--noconfirm", "--quiet"],
        "install": ["sudo", "pacman", "-S", "--noconfirm", "--quiet"],
        "clean": ["sudo", "pacman", "-Scc", "--noconfirm", "--quiet"],
    },
    "yum": {
        "is_installed": ["rpm", "-q"],
        "update": ["sudo", "yum", "makecache", "--assumeyes", "--quiet"],
        "upgrade": ["sudo", "yum", "upgrade", "--assumeyes", "--quiet"],
        "install": ["sudo", "yum", "install", "--assumeyes", "--quiet"],
        "clean": ["sudo", "yum", "clean", "all", "--quiet"],
    },
}


def get_supported() -> List[str]:
    return list(_PACKAGE_MANAGERS.keys())


def get_installed() -> List[str]:
    return [package_manager for package_manager in get_supported() if shell.is_installed(package_manager)]


def get_manager() -> str:
    try:
        return get_installed()[0]
    except IndexError:
        raise ValueError("No package manager available")


def package_is_installed(manager: str, package: str) -> bool:
    _validate(package_manager=manager)
    return shell.exit_0(cast(List[Union[str, Path]], _PACKAGE_MANAGERS[manager]["is_installed"] + [package]))


def update(manager: str) -> shell.SubprocessResult:
    _validate(package_manager=manager)
    return _run_command(_PACKAGE_MANAGERS[manager]["update"].copy())


def upgrade(manager: str, package: str) -> shell.SubprocessResult:
    _validate(package_manager=manager)
    try:
        return _run_command(_PACKAGE_MANAGERS[manager]["upgrade"] + [package])
    except CommandFailed as e:
        raise PackageUpgradeFailed(
            package_manager=manager,
            package=package,
            reason=e,
        ) from e


def install(manager: str, package: str) -> shell.SubprocessResult:
    _validate(package_manager=manager)
    try:
        return _run_command(_PACKAGE_MANAGERS[manager]["install"] + [package])
    except CommandFailed as e:
        raise PackageInstallationFailed(
            package_manager=manager,
            package=package,
            reason=e,
        ) from e


def clean(manager: str) -> shell.SubprocessResult:
    _validate(package_manager=manager)
    return _run_command(_PACKAGE_MANAGERS[manager]["clean"].copy())


def _run_command(command: Union[str, List[str]]) -> shell.SubprocessResult:
    return shell.run_command(
        command=cast(List[Union[str, Path]], command),
        verbose=(False, False),
        show_spinner=True,
    )


def _validate(package_manager: str) -> None:
    if package_manager not in get_supported():
        raise ValueError(f"Package manager `{package_manager}` not supported.")

    if package_manager not in get_installed():
        raise ValueError(f"Package manager `{package_manager}` not installed.")
