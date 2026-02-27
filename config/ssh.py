import copy
from dataclasses import dataclass
from pathlib import Path
from typing import cast

##### config start #####
_HOST: str | None = None
_PORT_DEBUG: int = 5678
_SSH_CONFIG_PATH: Path = Path.home() / ".ssh" / "config"
##### config end #####

_SSH_CONFIG = None


@dataclass
class SSHConfig:
    host_name: str | None = None
    port: int = 22
    user: str | None = None
    identity_file: Path | None = None


def get_config() -> SSHConfig:
    import helper.file as file

    global _SSH_CONFIG
    if _SSH_CONFIG:
        return copy.deepcopy(_SSH_CONFIG)

    ssh_config = SSHConfig()

    if not _HOST:
        raise ValueError("Host not specified.")

    if not _SSH_CONFIG_PATH.exists():
        raise FileNotFoundError(f"SSH config file not found: {_SSH_CONFIG_PATH}")

    found = False
    for line in file.read_lines(_SSH_CONFIG_PATH):
        line = line.strip()
        line_ = line.split()

        if line.startswith("#") or len(line_) < 2:
            continue

        line_[0] = line_[0].lower()

        if line_[0] == "host":
            if line_[1].lower() == _HOST.lower():
                ssh_config = SSHConfig()
                found = True
            elif found:
                found = False
        elif found:
            if line_[0] == "hostname":
                ssh_config.host_name = line_[1]
            elif line_[0] == "port":
                ssh_config.port = int(line_[1])
            elif line_[0] == "user":
                ssh_config.user = line_[1]
            elif line_[0] == "identityfile":
                ssh_config.identity_file = Path(line_[1]).expanduser()

    failed = []

    if not ssh_config.host_name:
        failed.append("`HostName`")
    if not ssh_config.user:
        failed.append("`User`")
    if not ssh_config.identity_file:
        failed.append("`IdentityFile`")

    if failed:
        raise RuntimeError(
            f"Failed to parse SSH config file: {_SSH_CONFIG_PATH}. Missing field(s): {', '.join(failed)}."
        )

    _SSH_CONFIG = ssh_config

    return copy.deepcopy(_SSH_CONFIG)


def get_host_name() -> str:
    return cast(str, get_config().host_name)


def get_port() -> int:
    return get_config().port


def get_user() -> str:
    return cast(str, get_config().user)


def get_identity_file() -> Path:
    return cast(Path, get_config().identity_file)


def get_port_debug() -> int:
    return _PORT_DEBUG
