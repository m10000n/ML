import os
import re
import stat
import sys
import time
from contextlib import nullcontext

from config import ssh as ssh_config
from helper import file, path, process, shell, system
from helper.clt.exception import CLTError, CommandFailed
from helper.print import Spinner, print_end, print_error, print_info, print_start
from helper.process import PROCESS_LOCK, Process

##### config start #####
_TTL_SSH_KEY_HOURS = 24
_KNOWN_HOSTS_PATH = path.home() / ".ssh" / "known_hosts"
##### config end #####

_TTL_SSH_KEY_SECONDS = _TTL_SSH_KEY_HOURS * 3600
_TIMEOUT = 2

_PROCESS_IDENTIFIER = "SSH TUNNEL"
_PROCESS_ID = process.get_reserved_id("ssh")


def __setup() -> None:
    print_start("Start SSH setup.")
    success = True

    try:
        __add_host()
    except CommandFailed as e:
        print_error(e.get_clt_message())
        success = False

    try:
        __add_key()
    except CLTError as e:
        print_error(e.get_clt_message())
        success = False

    end_message = "SSH setup completed."
    if success:
        print_end(end_message)
    else:
        print_error(end_message)


def __add_host() -> None:
    host_name = ssh_config.get_host_name()

    if not shell.exit_0(["ssh-keygen", "-F", host_name]):
        try:
            host_keys = shell.run_command(command=["ssh-keyscan", "-H", host_name], verbose=(False, False)).get_stdout()
        except CommandFailed as e:
            e.prepend_clt_message("Failed to add host to known hosts. Failed to retrieve host's public SSH key.")

        if _KNOWN_HOSTS_PATH.exists():
            file.append(path=_KNOWN_HOSTS_PATH, text=str(host_keys))
        else:
            file.write(path=_KNOWN_HOSTS_PATH, text=str(host_keys))

        print_info("Added host to known hosts.")


def _check_key_encrypted() -> None:
    key = file.read_lines(ssh_config.get_identity_file())
    header = key[0].strip()

    encrypted = False
    if header == "-----BEGIN OPENSSH PRIVATE KEY-----":
        for line in key[1:-1]:
            if "YmNyeXB0" in line:
                encrypted = True
                break
    elif re.match(r"^-----BEGIN (.+) PRIVATE KEY-----$", header):
        for line in key[1:-1]:
            if "ENCRYPTED" in line:
                encrypted = True
                break
    else:
        raise ValueError("Could not determine if your SSH key is encrypted (Supported key formats: PEM, OpenSSH).")

    if not encrypted:
        raise ValueError("Your SSH key is not encrypted.")


def __add_key() -> None:
    error_message = "Failed to add your SSH key to the ssh-agent."
    identity_file = ssh_config.get_identity_file()

    if not _agent_is_available():
        raise CLTError(message="SSH agent is not available.", clt_message=error_message)
    else:
        try:
            _check_key_encrypted()
        except ValueError as e:
            raise CLTError(message=e.args[0], clt_message=error_message) from e

        try:
            result = shell.run_command(command=["ssh-keygen", "-lf", identity_file], verbose=(False, False))
            fingerprint = result.get_stdout(as_list=True)[0]
        except CommandFailed as e:
            e.prepend_clt_message(f"{error_message}. Failed to get fingerprint of your SSH key.")

        try:
            fingerprints = shell.run_command(command=["ssh-add", "-l"], verbose=(False, False)).get_stdout(as_list=True)
        except CommandFailed:
            fingerprints = []

        added = False

        while not fingerprint in fingerprints:
            added = True

            try:
                shell.run_command(
                    [
                        "ssh-add",
                        "-t",
                        str(_TTL_SSH_KEY_SECONDS),
                        ssh_config.get_identity_file(),
                    ]
                )
                fingerprints = shell.run_command(command=["ssh-add", "-l"], verbose=(False, True)).get_stdout()
            except CommandFailed:
                pass

        if added:
            print_info("Added your SSH key to the ssh-agent.")


def __start_tunnel() -> None:
    tunnel_config = _get_tunnel_config()
    debug_port = ssh_config.get_port_debug()
    error_message = f"Failed to start a tunnel: {tunnel_config}."

    if _tunnel_is_active():
        print_error(f"{error_message} Tunnel is already active.")
        sys.exit(1)
    elif system.port_is_in_use(port=debug_port):
        print_error(f"{error_message} Local port is already in use.")
        sys.exit(1)
    else:
        __setup()

        process_ = shell.run_command_background(
            [
                "ssh",
                "-o",
                "ExitOnForwardFailure=yes",
                "-o",
                f"ConnectTimeout={_TIMEOUT}",
                "-i",
                ssh_config.get_identity_file(),
                "-L",
                f"{ssh_config.get_port_debug()}:localhost:{ssh_config.get_port_debug()}",
                "-N",
                f"{ssh_config.get_user()}@{ssh_config.get_host_name()}",
            ]
        )

        spinner = Spinner()
        spinner.start()
        time.sleep(_TIMEOUT + 0.2)
        spinner.stop()

        if process_.poll():
            print_error(error_message)
            if process_.stderr:
                print(process_.stderr.read(), end="")
            sys.exit(1)
        else:
            Process(
                pid=process_.pid,
                id_=_PROCESS_ID,
                command=_PROCESS_IDENTIFIER,
                meta=f"config: {tunnel_config}",
                important=False,
            )
            print_info(f"Started a tunnel: {tunnel_config}.")


def __stop_tunnel() -> None:
    with PROCESS_LOCK:
        if _tunnel_is_active(with_lock=False):
            process_ = process.get(id_=_PROCESS_ID, with_lock=False)
            config = (
                None
                if process_.meta is None
                else next((meta for meta in process_.meta if meta.startswith("config: ")), None)
            )

            if config is None:
                raise AssertionError("Tunnel is active but configuration is not set. This should never happen.")

            process_.kill(with_lock=False)
            print_info(f"Stopped tunnel: {config[len("config: "):]}.")
        else:
            print_error("Tunnel is not active.")
            sys.exit(1)


def __tunnel_info() -> None:
    if _tunnel_is_active():
        print_info(f"Tunnel is active: {_get_tunnel_config()}.")
    else:
        print_info("Tunnel is not active.")


def _get_tunnel_config() -> str:
    debug_port = ssh_config.get_port_debug()
    return f"{debug_port}:{ssh_config.get_host_name()}:{debug_port}"


def _tunnel_is_active(with_lock: bool = True) -> bool:
    with PROCESS_LOCK if with_lock else nullcontext():
        if process.is_tracked(id_=_PROCESS_ID, with_lock=False):
            process_ = process.get(id_=_PROCESS_ID, with_lock=False)
            config = (
                None
                if process_.meta is None
                else next((meta for meta in process_.meta if meta.startswith("config: ")), None)
            )

            if config is None:
                raise AssertionError("Tunnel is active but configuration is not set. This should never happen.")

            if _get_tunnel_config() != config[len("config: ") :]:
                process_.kill(with_lock=False)
                return False
            else:
                return True
        else:
            return False


def _agent_is_available() -> bool:
    ssh_auth_sock = os.environ.get("SSH_AUTH_SOCK")
    if not ssh_auth_sock:
        return False

    if not os.path.exists(ssh_auth_sock):
        return False

    mode = os.stat(ssh_auth_sock).st_mode
    if not stat.S_ISSOCK(mode):
        return False

    try:
        shell.run_command(command=["ssh-add", "-l"], verbose=(False, False))
        return True
    except CommandFailed as e:
        if "The agent has no identities" in e.get_stdout():
            return True
        else:
            return False
