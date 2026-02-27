# this file is used before the python environment is ready

import os
from typing import Dict, Union

from helper import bash, env
from helper import package_manager as package_manager_
from helper import path, shell, system
from helper.clt.exception import (
    CommandFailed,
    PackageError,
    PackageInstallationFailed,
    PackageUpgradeFailed,
)
from helper.print import print_end, print_error, print_info, print_start, print_warning

_PACKAGES = ["cloc", "direnv", "htop", "micromamba", "rsync", "sensors", "tmux", "watch"]
_CLOC_VERSION = "2.04"

_BINARIES_DIR = "/usr/local/bin"

_MAMBA_EXE = _BINARIES_DIR + "/micromamba"
_MAMBA_ROOT_PREFIX = path.home() / "micromamba"

_ENV = os.environ.copy()
_ENV["PATH"] = f"{_BINARIES_DIR}:$PATH" if _BINARIES_DIR not in _ENV["PATH"] else _ENV["PATH"]

_DEPENDENCIES_FOR_BINARIES: Dict[str, Union[bool, Union[Exception, None]]] = {
    "success": False,
    "exception": None,
}


def __main(install: bool) -> None:
    print_start(text="Start setup.", mode="primary")
    print()

    success = True

    __shell_config()
    print()

    if install:
        success = __install()
        print()

    env_success = env.__setup()
    success = success and env_success
    print()

    message = "Finished setup."
    if success:
        print_end(message, mode="primary")
    else:
        print_error(message, mode="primary")


def __shell_config() -> None:
    print_start(text="Start shell configuration.")
    if bash.add_ML():
        print_info(f"Added ML to `{bash.CONFIG_PATH}`.")
    print_end(text="Finished shell configuration.")


def __install() -> bool:
    print_start("Start installing packages")
    try:
        package_manager = package_manager_.get_manager()
    except ValueError:
        print_error("Missing package manager.")
        print_error(f"Supported package managers: {', '.join(package_manager_.get_supported())}.")
        exit(1)

    print_info(f"Package manager: {package_manager}")

    package_manager_.update(manager=package_manager)
    print_info("Updated package manager.")

    install_micromamba = True if "micromamba" in _PACKAGES else False
    install_direnv = True if "direnv" in _PACKAGES else False
    install_cloc = True if "cloc" in _PACKAGES else False

    packages = [package for package in _PACKAGES if package not in ["cloc", "direnv", "micromamba", "sensors"]]
    if system.get_system() == "linux":
        packages.append("sensors")

    install_failed = []
    update_failed = []

    for package in packages:
        try:
            __install_package(package_manager=package_manager, package=package)
        except PackageInstallationFailed:
            install_failed.append(package)
        except PackageUpgradeFailed:
            update_failed.append(package)

    if install_cloc:
        try:
            __install_cloc(package_manager=package_manager)
        except PackageInstallationFailed:
            install_failed.append("cloc")
        except PackageUpgradeFailed:
            update_failed.append("cloc")

    if install_direnv:
        try:
            __install_direnv(package_manager=package_manager)
        except PackageInstallationFailed:
            install_failed.append("direnv")
        except PackageUpgradeFailed:
            update_failed.append("direnv")

    if install_micromamba:
        try:
            __install_micromamba(package_manager=package_manager)
        except PackageInstallationFailed:
            install_failed.append("micromamba")
        except PackageUpgradeFailed:
            update_failed.append("micromamba")

    package_manager_.clean(manager=package_manager)

    if install_failed:
        print_error(f"Failed to install package(s): {', '.join(install_failed)}.")

    if update_failed:
        print_error(f"Failed to upgrade package(s): {', '.join(update_failed)}.")

    message = "Finished installing packages"
    if install_failed or update_failed:
        print_error(message)
        return False
    else:
        print_end(message)
        return True


def __install_package(package_manager: str, package: str) -> None:
    try:
        print_start(package)

        if package_manager_.package_is_installed(package_manager, package):
            package_manager_.upgrade(
                manager=package_manager,
                package=package,
            )
            print_end(f"Upgraded `{package}`.")
        elif shell.is_installed(package):
            print_warning(
                f"Cannot upgrade `{package}`. It is installed but is not managed by `{package_manager}`. `{package}` version: {shell.get_version(package)}"
            )
        else:
            package_manager_.install(
                manager=package_manager,
                package=package,
            )
            print_end(f"Installed `{package}`.")
    except PackageError as e:
        print_error(e.get_clt_message())
        raise


def __install_cloc(package_manager: str) -> None:
    try:
        print_start("cloc")

        if package_manager_.package_is_installed(package_manager, "cloc"):
            package_manager_.upgrade(
                manager=package_manager,
                package="cloc",
            )
            print_end("Upgraded `cloc`.")
        elif shell.is_installed("cloc"):
            print_warning(
                f"Cannot upgrade `cloc`. It is installed but is not managed by `{package_manager}`. `cloc` version: {shell.get_version('cloc')}"
            )
        else:
            try:
                package_manager_.install(
                    manager=package_manager,
                    package="cloc",
                )
                print_end("Installed `cloc`.")
            except PackageInstallationFailed:
                print_info("Failed to install `cloc` via package manager. Trying to install from binaries.")
                __install_cloc_from_binaries(package_manager=package_manager)
                print_end("Installed `cloc` from binaries.")
    except PackageError as e:
        print_error(e.get_clt_message())
        raise


def __install_direnv(package_manager: str) -> None:
    try:
        print_start("direnv")

        if package_manager_.package_is_installed(package_manager, "direnv"):
            package_manager_.upgrade(
                manager=package_manager,
                package="direnv",
            )
            message = (print_end, "Upgraded `direnv`.")
        elif shell.is_installed("direnv"):
            message = (
                print_warning,
                f"Cannot upgrade `direnv`. It is installed but is not managed by `{package_manager}`. `direnv` version: {shell.get_version('direnv')}",
            )
        else:
            try:
                package_manager_.install(
                    manager=package_manager,
                    package="direnv",
                )
                message = (print_end, "Installed `direnv`.")
            except PackageInstallationFailed:
                print_info("Failed to install `direnv` via package manager. Trying to install from binaries.")
                __install_direnv_from_binaries(package_manager=package_manager)
                message = (print_end, "Installed `direnv` from binaries.")

        shell.run_command(command=["direnv", "allow", path.project_root()], verbose=(False, False))

        if bash.add('eval "$(direnv hook bash)"\nexport DIRENV_LOG_FORMAT=""'):
            print_info(f"Added `direnv` hook to `{bash.CONFIG_PATH}`.")

        message[0](message[1])
    except PackageError as e:
        print_error(e.get_clt_message())
        raise


def __install_micromamba(package_manager: str) -> None:
    try:
        if shell.is_installed("mamba") and not shell.is_installed("micromamba"):
            print_info("Skipped installation of `micromamba` because `mamba` is installed.")
        else:
            print_start("micromamba")

            if shell.is_installed("micromamba"):
                try:
                    shell.run_command(
                        command=["micromamba", "self-update"],
                        verbose=(False, False),
                        show_spinner=True,
                    )
                    print_end("Upgraded `micromamba`.")
                except CommandFailed as e:
                    new_e = PackageUpgradeFailed(package="micromamba", package_manager=None, reason=e)
                    raise new_e from e
            else:
                env = os.environ.copy()
                env["MAMBA_ROOT_PREFIX"] = str(_MAMBA_ROOT_PREFIX)
                binary = False

                try:
                    package_manager_.install(manager=package_manager, package="micromamba")
                except PackageInstallationFailed as e:
                    print_info("Failed to install `micromamba` via package manager. Trying to install from binaries.")
                    __install_micromamba_from_binaries(package_manager=package_manager)
                    binary = True
                    env["MAMBA_EXE"] = _MAMBA_EXE

                try:
                    shell.run_command(
                        command=["micromamba", "shell", "init", "--shell", "bash"],
                        env=env,
                        verbose=(False, False),
                        show_spinner=True,
                    )
                    print_info("Initialized `micromamba`.")
                    print_end(f"Installed `micromamba` {'from binaries' if binary else f'via `{package_manager}`'}.")
                except CommandFailed as e:
                    new_e = PackageInstallationFailed(
                        package="micromamba",
                        package_manager=package_manager if not binary else None,
                        reason=e,
                    )  # type: ignore # this is a mypy bug
                    new_e.append_clt_message("Failed to initialize `micromamba`.")
                    raise new_e from e
    except PackageError as e:
        print_error(e.get_clt_message())
        raise


def __prepare_binary_installation(package_manager: str) -> None:
    if isinstance(_DEPENDENCIES_FOR_BINARIES["exception"], Exception):
        raise _DEPENDENCIES_FOR_BINARIES["exception"]
    elif not _DEPENDENCIES_FOR_BINARIES["success"]:
        packages = ["curl", "perl-core", "tar", "bzip2"]
        for package in packages:
            if package_manager_.package_is_installed(manager=package_manager, package=package):
                try:
                    package_manager_.upgrade(
                        manager=package_manager,
                        package=package,
                    )
                except CommandFailed:
                    pass
            elif not shell.is_installed(package):
                try:
                    package_manager_.install(
                        manager=package_manager,
                        package=package,
                    )
                except PackageInstallationFailed as e:
                    e.prepend_clt_message("Failed to prepare installation from binaries.")
                    _DEPENDENCIES_FOR_BINARIES["exception"] = e
                    raise

        os.makedirs(_BINARIES_DIR, exist_ok=True)

        path = os.environ.get("PATH", "")
        if not _BINARIES_DIR in path.split(":"):
            bash.add(f'export PATH="{_BINARIES_DIR}:$PATH"')
            print_info(f"Added `{_BINARIES_DIR}` to PATH in `{bash.CONFIG_PATH}`.")

        _DEPENDENCIES_FOR_BINARIES["success"] = True


def __install_cloc_from_binaries(package_manager: str) -> None:
    try:
        __prepare_binary_installation(package_manager=package_manager)

        clock_path_ = f"{_BINARIES_DIR}/cloc"
        shell.run_command(
            command=[
                "sudo",
                "wget",
                "-q",
                "-o",
                "/dev/null",
                f"https://github.com/AlDanial/cloc/releases/download/v{_CLOC_VERSION}/cloc-{_CLOC_VERSION}.pl",
                "-O",
                clock_path_,
            ],
            verbose=(False, False),
            show_spinner=True,
        )
        shell.run_command(command=["sudo", "chmod", "+x", clock_path_], verbose=(False, False))

    except (PackageInstallationFailed, CommandFailed) as e:
        print(e)
        new_e = PackageInstallationFailed(package="cloc", package_manager=None, reason=e)

        if isinstance(e, PackageInstallationFailed):
            new_e.append_clt_message(e.get_clt_message())

        raise new_e from e


def __install_direnv_from_binaries(package_manager: str) -> None:
    try:
        __prepare_binary_installation(package_manager=package_manager)

        env = _ENV.copy()
        env["bin_path"] = _BINARIES_DIR

        shell.run_command(
            command="curl -sfL https://direnv.net/install.sh | sudo -E bash",
            env=env,
            verbose=(False, False),
            show_spinner=True,
        )
    except (PackageInstallationFailed, CommandFailed) as e:
        new_e = PackageInstallationFailed(package="direnv", package_manager=None, reason=e)

        if isinstance(e, PackageInstallationFailed):
            new_e.append_clt_message(e.get_clt_message())

        raise new_e from e


def __install_micromamba_from_binaries(package_manager: str) -> None:
    try:
        __prepare_binary_installation(package_manager=package_manager)

        system_ = system.get_system()
        machine = system.get_machine()

        download_url = ""
        if system_ == "linux":
            if machine in ("x86_64", "amd64"):
                download_url = "https://micro.mamba.pm/api/micromamba/linux-64/latest"
            elif machine in ("aarch64", "arm64"):
                download_url = "https://micro.mamba.pm/api/micromamba/linux-aarch64/latest"
            elif machine == "ppc64le":
                download_url = "https://micro.mamba.pm/api/micromamba/linux-ppc64le/latest"

        elif system_ == "darwin":
            if machine == "x86_64":
                download_url = "https://micro.mamba.pm/api/micromamba/osx-64/latest"
            elif machine == "arm64":
                download_url = "https://micro.mamba.pm/api/micromamba/osx-arm64/latest"

        if not download_url:
            raise RuntimeError(f"Unsupported system: {system_} and machine: {machine}")

        shell.run_command(
            command=f"sudo curl -Ls '{download_url}' | sudo tar -xvj --strip-components=1 -C '{_BINARIES_DIR}' bin/micromamba",
            verbose=(False, False),
            show_spinner=True,
        )
    except (PackageInstallationFailed, RuntimeError, CommandFailed) as e:
        new_e = PackageInstallationFailed(package="micromamba", package_manager=None, reason=e)

        if isinstance(e, PackageInstallationFailed):
            new_e.append_clt_message(e.get_clt_message())
        if isinstance(e, RuntimeError):
            new_e.append_clt_message(e.args[0])

        raise new_e from e
