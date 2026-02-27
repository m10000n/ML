# this file is used before the python environment is ready

import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Dict, List, Tuple, Union

from helper import bash
from helper.print import Spinner


class SubprocessResult:
    def __init__(
        self,
        command: Union[str, List[Union[str]]],
        exit_code: int,
        out: List[Tuple[str, int]],
    ):
        self._command = command
        self._exit_code = exit_code
        self._out = out

    def get_command(self) -> Union[str, List[Union[str]]]:
        return self._command

    def get_exit_code(self) -> int:
        return self._exit_code

    def get_out(self, as_list: bool = False) -> Union[str, List[str]]:
        out = [line for line, _ in self._out]
        return out if as_list else "\n".join(out)

    def get_stdout(self, as_list: bool = False) -> Union[str, List[str]]:
        stdout = [line for line, _ in self._out if _ == 1]
        return stdout if as_list else "\n".join(stdout)

    def get_stderr(self, as_list: bool = False) -> Union[str, List[str]]:
        stderr = [line for line, _ in self._out if _ == 2]
        return stderr if as_list else "\n".join(stderr)

    def was_successful(self) -> bool:
        return self._exit_code == 0


def run_command(
    command: Union[str, List[Union[str, Path]]],
    cwd: Union[str, Path, None] = None,
    env: Union[Dict[str, str], None] = None,
    verbose: Tuple[bool, bool] = (True, True),
    show_spinner: bool = False,
) -> SubprocessResult:
    from helper.clt.exception import CommandFailed

    command_ = command if isinstance(command, str) else [str(c) for c in command]

    if show_spinner and verbose[0]:
        raise ValueError("It does not make sense to show a spinner when writing to stdout.")
    shell = True if isinstance(command_, str) else False
    executable = bash.BINARIES if shell else None

    try:
        process = subprocess.Popen(
            command_,
            shell=shell,
            executable=executable,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        out: List[Tuple[str, int]] = []

        spinner = None
        if show_spinner:
            spinner = Spinner()
            spinner.start()

        # process.stdout is not None
        def read_stdout() -> None:
            for line in iter(process.stdout.readline, ""):  # type: ignore
                line = line.rstrip("\n")
                out.append((line, 1))

                if verbose[0]:
                    if spinner:
                        spinner.stop()
                    print(line)

            process.stdout.close()  # type: ignore

        # process.stderr is not None
        def read_stderr() -> None:
            for line in iter(process.stderr.readline, ""):  # type: ignore
                line = line.rstrip("\n")
                out.append((line, 2))

                if verbose[1]:
                    if spinner:
                        spinner.stop()
                    print(line)

            process.stderr.close()  # type: ignore

        stdout_thread = threading.Thread(target=read_stdout)
        stderr_thread = threading.Thread(target=read_stderr)

        stdout_thread.start()
        stderr_thread.start()

        stdout_thread.join()
        stderr_thread.join()

        process.wait()

        if spinner:
            spinner.stop()

        result = SubprocessResult(command=command_, exit_code=process.returncode, out=out)

        if process.returncode != 0:
            raise CommandFailed(subprocess_result=result)
        else:
            return result
    except FileNotFoundError:
        result = SubprocessResult(
            command=command_,
            exit_code=127,
            out=[(f"{command_[0]}: command not found", 2)],
        )
        raise CommandFailed(result)


def run_command_std(
    command: Union[str, List[Union[str, Path]]],
    cwd: Union[str, Path, None] = None,
    env: Union[Dict[str, str], None] = None,
) -> SubprocessResult:
    from helper.clt.exception import CommandFailed

    command_ = command if isinstance(command, str) else [str(c) for c in command]

    shell = True if isinstance(command_, str) else False
    executable = bash.BINARIES if shell else None

    try:
        process = subprocess.run(
            command_,
            shell=shell,
            executable=executable,
            cwd=cwd,
            env=env,
            stdout=sys.stdout,
            stderr=sys.stderr,
            text=True,
        )

        result = SubprocessResult(command=command_, exit_code=process.returncode, out=[])

        if process.returncode != 0:
            raise CommandFailed(result)
        else:
            return result

    except FileNotFoundError:
        result = SubprocessResult(
            command=command_,
            exit_code=127,
            out=[(f"{command_[0]}: command not found", 2)],
        )
        raise CommandFailed(result)


def run_command_background(
    command: Union[str, List[Union[str, Path]]],
    cwd: Union[str, Path, None] = None,
    env: Union[Dict[str, str], None] = None,
) -> subprocess.Popen:
    from helper.clt.exception import CommandFailed

    command_ = command if isinstance(command, str) else [str(c) for c in command]

    shell = True if isinstance(command_, str) else False
    executable = bash.BINARIES if shell else None

    try:
        return subprocess.Popen(
            command_,
            shell=shell,
            executable=executable,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    except FileNotFoundError:
        result = SubprocessResult(
            command=command_,
            exit_code=127,
            out=[(f"{command_[0]}: command not found", 2)],
        )
        raise CommandFailed(result)


def exit_0(
    command: Union[str, List[Union[str, Path]]],
    env: Union[Dict[str, str], None] = None,
) -> bool:
    command_ = command if isinstance(command, str) else tuple(str(c) for c in command)

    shell = True if isinstance(command_, str) else False
    executable = bash.BINARIES if shell else None

    try:
        subprocess.run(
            command_,
            shell=shell,
            executable=executable,
            env=env,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return False
    return True


def is_installed(tool: str) -> bool:
    return True if shutil.which(tool) else False


def get_version(tool: str) -> str:
    return run_command(command=[tool, "--version"], verbose=(False, False)).get_stdout(as_list=True)[0]


def is_in_path(dir: Union[str, Path]) -> bool:
    paths = [str(Path(path).resolve()) for path in os.environ.get("PATH", "").split(":")]
    return str(Path(dir).resolve()) in paths
