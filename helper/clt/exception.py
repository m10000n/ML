# this file is used before the python environment is ready

from abc import ABCMeta
from typing import List, Union

from helper.shell import SubprocessResult


class CLTError(Exception, metaclass=ABCMeta):
    def __init__(self, message: str, clt_message: str = ""):
        super().__init__(message)
        self._message = message
        self._clt_messages = [clt_message, message] if clt_message else [message]

    def get_message(self) -> str:
        return self._message

    def get_clt_message(self) -> str:
        return " ".join(self._clt_messages)

    def prepend_clt_message(self, message: str) -> None:
        self._clt_messages.insert(0, message)

    def append_clt_message(self, message: str) -> None:
        self._clt_messages.append(message)


# shell command
class CommandFailed(CLTError):
    def __init__(self, subprocess_result: SubprocessResult):
        message = f"Command `{subprocess_result.get_command()}` failed."
        super().__init__(message=message)
        self._subprocess_result = subprocess_result

    def get_command(self) -> Union[str, List[str]]:
        return self._subprocess_result.get_command()

    def get_exit_code(self) -> int:
        return self._subprocess_result.get_exit_code()

    def get_out(self, as_list: bool = False) -> Union[str, List[str]]:
        return self._subprocess_result.get_out(as_list=as_list)

    def get_stdout(self, as_list: bool = False) -> Union[str, List[str]]:
        return self._subprocess_result.get_stdout(as_list=as_list)

    def get_stderr(self, as_list: bool = False) -> Union[str, List[str]]:
        return self._subprocess_result.get_stderr(as_list=as_list)

    def __str__(self) -> str:
        return "\n".join(
            [
                self._message,
                f"Exit code: {self._subprocess_result.get_exit_code()}",
                f"Output:\n{self._subprocess_result.get_stderr()}",
            ]
        )


# package
class PackageError(CLTError, metaclass=ABCMeta):
    def __init__(
        self,
        message: str,
        package: str,
        package_manager: Union[str, None],
        reason: Exception,
    ):
        super().__init__(message)
        self._package = package
        self._package_manager = package_manager
        self._reason = reason

    def get_package(self) -> str:
        return self._package

    def get_package_manager(self) -> Union[str, None]:
        return self._package_manager

    def get_reason(self) -> Exception:
        return self._reason

    def __str__(self) -> str:
        return "\n".join(
            [
                self._message,
                f"Package: {self._package}",
                f"Package manager: {self._package_manager}",
                f"Reason:\n {self._reason}",
            ]
        )


class PackageInstallationFailed(PackageError):
    def __init__(self, package: str, package_manager: Union[str, None], reason: Exception):
        message = f"Failed to install `{package}` {f'via `{package_manager}`' if package_manager else 'from binaries'}."
        super().__init__(
            message=message,
            package=package,
            package_manager=package_manager,
            reason=reason,
        )


class PackageUpgradeFailed(PackageError):
    def __init__(self, package: str, package_manager: Union[str, None], reason: Exception):
        message = f"Failed to {'' if package_manager else 'self-'}upgrade `{package}`{f' via `{package_manager}`' if package_manager else ''}."
        super().__init__(
            message=message,
            package=package,
            package_manager=package_manager,
            reason=reason,
        )


# tmux
class TmuxError(CLTError, metaclass=ABCMeta):
    def __init__(self, message: str, clt_message: str):
        super().__init__(message=message, clt_message=clt_message)


class TmuxSessionNotFound(TmuxError):
    def __init__(self, session_name: str, clt_message: str = ""):
        message = f"tmux session `{session_name}` does not exist."
        super().__init__(message=message, clt_message=clt_message)
        self._session_name = session_name

    def get_session_name(self) -> str:
        return self._session_name


class TmuxWindowNotFound(TmuxError):
    def __init__(self, window_name: str, clt_message: str = ""):
        message = f"tmux window `{window_name}` does not exist."
        super().__init__(message=message, clt_message=clt_message)
        self._window_name = window_name

    def get_window_name(self) -> str:
        return self._window_name


class TmuxSessionAlreadyExists(TmuxError):
    def __init__(self, session_name: str, clt_message: str = ""):
        message = f"tmux session `{session_name}` already exists."
        super().__init__(message=message, clt_message=clt_message)
        self._session_name = session_name

    def get_session_name(self) -> str:
        return self._session_name


class TmuxWindowAlreadyExists(TmuxError):
    def __init__(self, window_name: str, clt_message: str = ""):
        message = f"tmux window `{window_name}` already exists."
        super().__init__(message=message, clt_message=clt_message)
        self._window_name = window_name

    def get_window_name(self) -> str:
        return self._window_name


class TmuxPaneNotFound(TmuxError):
    def __init__(self, window_name: str, pane_index: int, clt_message: str = ""):
        message = f"tmux pane `{window_name}.{pane_index}` does not exist."
        super().__init__(message=message, clt_message=clt_message)
        self._window_name = window_name
        self._pane_index = pane_index

    def get_window_name(self) -> str:
        return self._window_name

    def get_pane_index(self) -> int:
        return self._pane_index


class TmuxMultiplePanes(TmuxError):
    def __init__(self, window_name: str, clt_message: str = ""):
        message = f"tmux window `{window_name}` has multiple panes."
        super().__init__(message=message, clt_message=clt_message)
        self._window_name = window_name

    def get_window_name(self) -> str:
        return self._window_name


class TmuxWindowNameInvalid(TmuxError):
    def __init__(self, window_name: str, reason: str, clt_message: str = ""):
        message = f"tmux window name `{window_name}` is invalid. {reason}"
        super().__init__(message=message, clt_message=clt_message)
        self._window_name = window_name

    def get_window_name(self) -> str:
        return self._window_name
