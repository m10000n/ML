import os
import re
import sys
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Literal, ParamSpec, TypeVar

import libtmux

from config import file_names
from helper import debug, env, file, path, process
from helper.clt.exception import (
    TmuxMultiplePanes,
    TmuxPaneNotFound,
    TmuxSessionAlreadyExists,
    TmuxSessionNotFound,
    TmuxWindowAlreadyExists,
    TmuxWindowNameInvalid,
    TmuxWindowNotFound,
)
from helper.print import print_error, print_info

##### config start #####
_SESSION_NAME = "ML"
_WINDOW_NAME_0 = "home"
##### config end #####

_TMUX_CONFIG_PATH = path.config(absolute=True) / "tmux.conf"
_TMUX_DIR = path.tmp(absolute=True) / "tmux"

_SERVER = libtmux.Server()

_WINDOW_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_]+$")
_WINDOW_NAME_PATTERN_INTERNAL = re.compile(r"^[A-Za-z0-9_-]+$")


_P = ParamSpec("_P")
_R = TypeVar("_R")

wrapper_flag: ContextVar[bool] = ContextVar("wrapper_flag", default=False)


def session_exists() -> bool:
    try:
        _get_session()
        return True
    except TmuxSessionNotFound:
        return False


def window_exists(name: str) -> bool:
    try:
        _get_window(name=name)
        return True
    except (TmuxSessionNotFound, TmuxWindowNotFound):
        return False


def new_session() -> libtmux.Session:
    _check_window_name(name=_SESSION_NAME, internal=False)
    try:
        session = _SERVER.new_session(
            session_name=_SESSION_NAME,
            window_name=_WINDOW_NAME_0,
        )
        pane = session.windows[0].panes[0]
        _pane_setup(pane)
        return session
    except libtmux.exc.LibTmuxException as e:
        raise TmuxSessionAlreadyExists(
            session_name=_SESSION_NAME,
            clt_message="Failed to create a new tmux session.",
        ) from e


def new_window(name: str) -> libtmux.Window:
    return _new_window(name=name, internal=False)


def new_pane(window_name: str) -> libtmux.Pane:
    try:
        window = _get_window(name=window_name)
        pane = window.split_window()
        _pane_setup(pane)
        return pane
    except (TmuxSessionNotFound, TmuxWindowNotFound) as e:
        e.prepend_clt_message("Failed to create a new tmux pane.")
        raise


def kill_session() -> None:
    try:
        _get_session().kill_session()
    except TmuxSessionNotFound as e:
        e.prepend_clt_message("Failed to kill tmux session.")
        raise


def kill_window(name: str, force: bool = False) -> None:
    message = "Failed to kill tmux window."

    try:
        window = _get_window(name=name)
        if len(window.panes) > 1 and not force:
            raise TmuxMultiplePanes(window_name=name, clt_message=message)
        else:
            window.kill_window()
    except (TmuxSessionNotFound, TmuxWindowNotFound) as e:
        e.prepend_clt_message(message)
        raise


def kill_pane(window_name: str, index: int = 0) -> None:
    try:
        pane = _get_pane(window_name=window_name, index=index)
        pane.kill()
    except (TmuxSessionNotFound, TmuxWindowNotFound, TmuxPaneNotFound) as e:
        e.prepend_clt_message("Failed to kill tmux pane.")
        raise


@process.not_important
def attach(window_name: str | None = None) -> None:
    try:
        if window_name:
            _get_window(name=window_name)
        else:
            session = _get_session()
            window_name = session.windows[0].name

        if os.environ.get("TMUX"):
            _SERVER.cmd("switch-client", "-t", f"{_SESSION_NAME}:{window_name}")
        else:
            _SERVER.cmd("attach-session", "-t", f"{_SESSION_NAME}:{window_name}")

        sys.exit(0)
    except (TmuxSessionNotFound, TmuxWindowNotFound) as e:
        e.prepend_clt_message("Failed to attach.")
        raise


def detach_all() -> None:
    for client in _SERVER.sessions:
        attached = client.session_attached_list
        if attached:
            for attached_ in attached.split(","):
                _SERVER.cmd("detach-client", "-t", attached_)


def clear(pane: libtmux.Pane, clear_history: bool) -> None:
    send_command(
        pane=pane,
        command=f"clear && tmux clear-history -t {pane.pane_id}; {'history -c' if clear_history else 'history -d $((HISTCMD - 1))'}",
    )


# `func` arguments (args, kwargs) must be primitive-type (int, float, bool, str).
def run(
    window_name: str,
    func: Callable,
    process_id: str | None = None,
    process_command: str | None | Literal["parent"] = None,
    process_meta: list[str] | str | None | Literal["parent"] = None,
    process_important: bool | None | Literal["parent"] = None,
    attach_if_exists: bool = False,
    pane: bool = False,
    write_log: bool = True,
    attach: bool = False,
    *args: Any,
    **kwargs: Any,
) -> libtmux.Pane | None:
    from helper.tmux import attach as attach_

    if not pane and "-" in window_name:
        raise ValueError(f"If `pane` is False, `window_name` ({window_name}) must not contain '-'.")

    if debug.is_active():
        func(*args, **kwargs)
        return None

    if attach_if_exists and window_exists(window_name):
        attach_(window_name=window_name)
        return None

    try:
        _get_session()
    except TmuxSessionNotFound:
        new_session()

    if pane and window_exists(window_name):
        pane_ = new_pane(window_name)
    else:
        try:
            window_names = [str(window.name) for window in _get_windows(window_name)]
            max_idx = max(
                [int(window_name.split("-")[-1]) if "-" in window_name else 0 for window_name in window_names]
            )
            window_name = f"{window_name}-{max_idx + 1}"
            pane_ = _new_window(name=window_name, internal=True).panes[0]
        except TmuxWindowNotFound:
            pane_ = _new_window(name=window_name, internal=False).panes[0]

    original_func = func
    while hasattr(original_func, "__wrapped__"):
        original_func = original_func.__wrapped__

    module = path.make_module(Path(original_func.__code__.co_filename))
    current_module = path.make_module(path.file_path())

    process_ = process.get_this()

    process_command_ = process_.command if process_command == "parent" else process_command
    if process_meta == "parent":
        process_meta_ = process_.meta
    elif isinstance(process_meta, str):
        process_meta_ = [process_meta]

    process_important_ = process_.important if process_important == "parent" else process_important

    if process_meta is None:
        process_meta_ = process_.meta
    elif isinstance(process_meta, str):
        process_meta_ = [process_meta]
    else:
        process_meta_ = process_meta

    run_command = [
        repr(process_id),
        repr(process_command_),
        repr(None if process_meta_ is None else ",".join(process_meta_)),
        repr(process_important_),
        str(write_log),
        f"{original_func.__name__}{'.__wrapped__' if wrapper_flag.get() else ''}",
    ]
    run_command.extend(repr(arg) for arg in args)
    run_command.extend(f"{key}={repr(value) if isinstance(value, str) else value}" for key, value in kwargs.items())
    run_command_ = ", ".join(run_command)

    python_command = [
        f"from {current_module} import _run_tracked",
        f"from {module} import {original_func.__name__}",
        f"_run_tracked({run_command_})",
    ]
    python_command_ = "; ".join(python_command)
    python_command_ = f"python -c {repr(python_command_)}"

    clear_command = [
        "clear",
        f"tmux clear-history -t {_SESSION_NAME}:{window_name}.{pane_.index}",
    ]
    clear_command_ = " && ".join(clear_command)
    command = f"{clear_command_} && {python_command_} && exit"
    send_command(pane=pane_, command=command)

    if attach:
        attach_(window_name=window_name)

    return pane_


def send_command(command: str, pane: libtmux.Pane | None = None) -> None:
    if not pane:
        pane = _get_this_pane()
    pane.send_keys(cmd=command, suppress_history=True)


def send_commands(pane: libtmux.Pane, commands: list[str]) -> None:
    for command in commands:
        send_command(pane=pane, command=command)


def write_log() -> None:
    try:
        pane = _get_this_pane()
    except RuntimeError as e:
        raise RuntimeError(f"Failed to write log. {e}") from e

    os.makedirs(_TMUX_DIR, exist_ok=True)
    log_file_path = _TMUX_DIR / file_names.TMUX_LOG_FILE_NAME.format(window=pane.window.name, pane=pane.index)

    output = pane.capture_pane(start=-10000)

    if isinstance(output, str):
        file.write(path=log_file_path, text=output, overwrite=True, lock=True)
    else:
        file.write_lines(path=log_file_path, lines=output, overwrite=True, lock=True)


def _get_session() -> libtmux.Session:
    session = next((session for session in _SERVER.sessions if session.name == _SESSION_NAME), None)
    if session:
        return session
    else:
        raise TmuxSessionNotFound(session_name=_SESSION_NAME)


def _get_window(name: str) -> libtmux.Window:
    session = _get_session()
    window = next((window for window in session.windows if window.name == name), None)
    if window:
        return window
    else:
        raise TmuxWindowNotFound(window_name=name)


def _get_windows(name: str) -> list[libtmux.Window]:
    session = _get_session()
    windows = [window for window in session.windows if str(window.name).startswith(name)]
    if windows:
        return windows
    else:
        raise TmuxWindowNotFound(window_name=name)


def _get_pane(window_name: str, index: int) -> libtmux.Pane:
    try:
        window = _get_window(name=window_name)
        return window.panes[index]
    except IndexError:
        raise TmuxPaneNotFound(window_name=window_name, pane_index=index)


def _get_this_pane() -> libtmux.Pane:
    pane_id = os.getenv("TMUX_PANE")
    if not pane_id:
        raise RuntimeError("This function must be called from within a tmux pane.")

    return next(pane for pane in _SERVER.panes if pane.pane_id == pane_id)


def _check_window_name(name: str, internal: bool) -> None:
    regex = _WINDOW_NAME_PATTERN_INTERNAL if internal else _WINDOW_NAME_PATTERN

    if not regex.fullmatch(name):
        reason = "It must contain only letters, numbers, '_'" + (", '-'" if internal else "") + "."
        reason = "It must only contain letters, numbers, "
        reason += "'_', and '-'." if internal else "and '_'."
    elif name.lower() == "session":
        reason = "You cannot name your window: {window_name}."
    else:
        reason = None

    if reason is not None:
        raise TmuxWindowNameInvalid(window_name=name, reason=reason)


def _new_window(name: str, internal: bool) -> libtmux.Window:
    _check_window_name(name=name, internal=internal)

    message = "Failed to create a new tmux window."

    try:
        session = _get_session()

        if name in [window.name for window in session.windows]:
            raise TmuxWindowAlreadyExists(window_name=name, clt_message=message)
        else:
            window = session.new_window(window_name=name, environment=dict(os.environ))
            _pane_setup(window.panes[0])
            return window
    except TmuxSessionNotFound as e:
        e.prepend_clt_message(message)
        raise


def _pane_setup(pane: libtmux.Pane) -> None:
    commands = [
        f"cd {path.project_root()}",
        f'tmux source-file "{_TMUX_CONFIG_PATH}"',
        # mamba needs to be activated twice to show the active environment in the prompt
        env.get_activate_command(),
        env.get_activate_command(),
        "history -c",
    ]
    send_commands(pane=pane, commands=commands)
    clear(pane=pane, clear_history=True)


def _run_tracked(
    process_id: str | None,
    process_command: str | None,
    process_meta: str | None,
    process_important: bool,
    write_log_: bool,
    func: Callable[_P, _R],
    *args: _P.args,
    **kwargs: _P.kwargs,
) -> None:
    process_ = None
    try:
        meta_ = [] if process_meta is None else process_meta.split(",")
        meta_ = [m for m in meta_ if not m.startswith("tmux_window:")]
        meta_.append(f"tmux_window: {str(_get_this_pane().window.name)}")
        process_ = process.Process(id_=process_id, command=process_command, meta=meta_, important=process_important)
        func(*args, **kwargs)
    finally:
        if write_log_:
            write_log()
        if process_:
            process_.untrack()


def __list_windows() -> None:
    try:
        session = _get_session()
        print("tmux windows:")
        for window in session.windows:
            print(f"\t{window.window_name}")
    except TmuxSessionNotFound as e:
        e.prepend_clt_message("Failed to list tmux windows.")
        print_error(e.get_clt_message())
        sys.exit(1)


def __new(window_name: str | None = None, force: bool = False) -> None:
    try:
        new_session()
        print_info("Created a new tmux session.")
    except TmuxSessionAlreadyExists as e:
        if window_name is None:
            print_error(e.get_clt_message())
            exit(1)
        else:
            pass

    if window_name:
        try:
            new_window(name=window_name)
            print_info(f"Created a new tmux window: {window_name}.")
        except (TmuxWindowAlreadyExists, TmuxWindowNameInvalid) as e:
            if isinstance(e, TmuxWindowAlreadyExists):
                if force:
                    new_pane(window_name=window_name)
                    print_info(f"Created a new pane in tmux window: {window_name}.")
                else:
                    print_error(e.get_clt_message())
                    print_error("Use --force to create a new pane.")
                    sys.exit(1)
            elif isinstance(e, TmuxWindowNameInvalid):
                print_error(e.get_clt_message())
                sys.exit(1)


def __kill(target_name: str, pane: int | None = 0, force: bool = False) -> None:
    try:
        if target_name.lower() == "session":
            kill_session()
        elif pane is None:
            kill_window(name=target_name, force=force)
        else:
            kill_pane(window_name=target_name, index=pane)
            print_info(f"Killed tmux pane: {target_name}.{pane}")
    except (
        TmuxSessionNotFound,
        TmuxWindowNotFound,
        TmuxPaneNotFound,
        TmuxMultiplePanes,
    ) as e:
        print_error(e.get_clt_message())
        if isinstance(e, TmuxMultiplePanes):
            print_error("Use --force to kill the tmux window anyway.")
        sys.exit(1)

    killed_session = target_name == "session"
    killed_window = target_name != "session" and pane is None

    if target_name != "session":
        try:
            _get_window(name=target_name)
        except (TmuxSessionNotFound, TmuxWindowNotFound) as e:
            killed_window = True
            if isinstance(e, TmuxSessionNotFound):
                killed_session = True

    if killed_window:
        print_info(f"Killed tmux window: {target_name}.")
    if killed_session:
        print_info("Killed tmux session.")


def __attach(window_name: str | None = None) -> None:
    try:
        attach(window_name=window_name)
    except (TmuxSessionNotFound, TmuxWindowNotFound) as e:
        print_error(e.get_clt_message())
        sys.exit(1)


def __clear() -> None:
    try:
        pane = _get_this_pane()
        clear(pane=pane, clear_history=False)
    except RuntimeError as e:
        print_error(f"Failed to clear tmux pane. {e}")
        sys.exit(1)


def __write_log() -> None:
    try:
        write_log()
    except RuntimeError as e:
        print_error(e.args[0])
        sys.exit(1)


# Must be first decorator.
# Function arguments must be primitive-type (int, float, bool, str).
def in_tmux(
    window_name: str,
    process_id: str | None = None,
    process_command: str | None | Literal["parent"] = None,
    process_meta: list[str] | str | None | Literal["parent"] = None,
    process_important: bool | None | Literal["parent"] = None,
    attach_if_exists: bool = False,
    pane: bool = False,
    write_log: bool = True,
    attach: bool = True,
) -> Callable[[Callable[_P, _R]], Callable[_P, None]]:
    def decorator(func: Callable[_P, _R]) -> Callable[_P, None]:
        @wraps(func)
        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> None:
            token = wrapper_flag.set(True)
            try:
                # no keyword arguments here
                run(
                    window_name,
                    func,
                    process_id,
                    process_command,
                    process_meta,
                    process_important,
                    attach_if_exists,
                    pane,
                    write_log,
                    attach,
                    *args,
                    **kwargs,
                )
            finally:
                wrapper_flag.reset(token)

        return wrapper

    return decorator
