# this file is used before the python environment is ready

import os
import platform
import random
import socket
import sys
from textwrap import dedent
from typing import Dict, List, Literal, Optional, Union, cast

from helper import file, path
from helper.print import print_error, print_info, print_warning

##### config start #####
_MAX_NUM_WORKERS_DOWNLOAD = 6
_MASTER_ADDR: str = "localhost"
_MASTER_PORT: int = 29500
_DEFAULT_SEED: int = 42
_DEFAULT_AUTOCAST: bool = True
_DEFAULT_PREFETCH_FACTOR: int = 1
##### config end #####

_SYSTEM_DIR = path.tmp(absolute=True) / "system"
_NUM_WORKERS_PATH = _SYSTEM_DIR / "num_workers"
_PU_PATH = _SYSTEM_DIR / "pu"
_SEED_PATH = _SYSTEM_DIR / "seed"
_AUTOCAST_PATH = _SYSTEM_DIR / "autocast"
_PREFETCH_FACTOR_PATH = _SYSTEM_DIR / "prefetch_factor"


def _removesuffix(s: str, suffix: str) -> str:
    """Python 3.8 compatible replacement for str.removesuffix()."""
    if suffix and s.endswith(suffix):
        return s[: -len(suffix)]
    return s


# set env variables
def init_system() -> None:
    set_num_workers()
    set_pu()
    set_seed()
    set_autocast()
    set_prefetch_factor()


# system
def __sytem_info() -> None:
    cpu_info = get_cpu_info()

    info = dedent(
        f"""\
        OS: {platform.system()}
        CPU:
            Brand: {cpu_info["brand"]}
            Architecture: {cpu_info["architecture"]}
            Physical cores: {cpu_info["physical_cores"]}
            Logical cores: {cpu_info["logical_cores"]}
        RAM: {get_ram_info()}
    """
    )

    gpu_info = get_gpu_info()
    if gpu_info is not None:
        n_gpus = len(gpu_info)

        for i, gpu in enumerate(gpu_info):
            info += dedent(
                f"""\
                GPU {gpu["id"]}:
                    Name: {gpu["name"]}
                    Memory: {gpu["memory"]}"""
            )
            info += "\n" if i < n_gpus - 1 else ""
    else:
        info += "GPU: None"

    print(info)


def get_system() -> Literal["linux", "darwin"]:
    system = platform.system().lower()
    if system not in ["linux", "darwin"]:
        raise RuntimeError(f"Unsupported OS: {platform.system()}")
    return cast(Literal["linux", "darwin"], system)


def get_machine() -> str:
    return platform.machine().lower()


def get_num_physical_cores() -> int:
    import psutil

    return psutil.cpu_count(logical=False)


def get_num_logical_cores() -> int:
    import psutil

    return psutil.cpu_count(logical=True)


def port_is_in_use(port: int, host: str = "localhost") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as socket_:
        try:
            socket_.bind((host, port))
            return False
        except socket.error:
            return True


def get_cpu_info() -> Dict[str, Union[str, int]]:
    import cpuinfo
    import psutil

    cpu_info = cpuinfo.get_cpu_info()
    return {
        "brand": str(cpu_info.get("brand_raw", "Unknown")),
        "architecture": str(cpu_info.get("arch", "Unknown")),
        "physical_cores": int(psutil.cpu_count(logical=False)),
        "logical_cores": int(psutil.cpu_count(logical=True)),
    }


def get_ram_info() -> str:
    import psutil

    from helper import shell

    if platform.system() == "Darwin":
        try:
            result = shell.run_command(["sysctl", "hw.memsize"], verbose=(False, False))
        except Exception as e:
            raise RuntimeError(f"Error fetching total RAM: {e}")

        total_bytes = int(cast(str, result.get_stdout()).split(":")[1].strip())
        total_ram = round(total_bytes / (1024**3), 2)
    else:
        memory_info = psutil.virtual_memory()
        total_ram = round(memory_info.total / (1024**3), 2)

    return f"{total_ram} GB"


def get_gpu_info() -> Union[List[Dict[str, Union[str, int]]], None]:
    import GPUtil

    try:
        gpus = GPUtil.getGPUs()
    except ValueError:
        gpus = []

    gpus_ = []
    for gpu in gpus:
        gpus_.append({"id": gpu.id, "name": gpu.name, "memory": f"{gpu.memoryTotal} MB"})

    return gpus_ if gpus_ else None


# number of workers
def __set_num_workers(num_workers: int) -> None:
    from helper import debug

    if num_workers < 1:
        raise ValueError(f"`num_workers` ({num_workers}) must be >= 1.")

    if debug.is_active():
        print_error("Setting the number of workers is disabled in debug mode.")
        return

    file.write(path=_NUM_WORKERS_PATH, text=str(num_workers), overwrite=True, lock=True)
    print_info(f"Number of workers set to `{num_workers}`.")
    num_physical_cores = get_num_physical_cores()
    if num_workers >= num_physical_cores:
        print_warning(
            f"Number of workers ({num_workers}) {'is equal to' if num_workers == num_physical_cores else 'is greater than'} the number of physical cores ({num_physical_cores}). This can cause problems."
        )


def __reset_num_workers() -> None:
    from helper import debug

    if debug.is_active():
        print_error("Resetting the number of workers is disabled in debug mode.")
        return

    if _NUM_WORKERS_PATH.exists():
        _NUM_WORKERS_PATH.unlink()
        print_info(f"Number of workers reset. Defaults to `{get_num_workers()}`.")
    else:
        print_error(f"Failed to reset number of workers. Number of workers is not set.")


def __num_workers_info() -> None:
    from helper import debug

    num_workers = get_num_workers()

    if debug.is_active():
        print_info(f"Debugging is active. Number of workers defaults to `{num_workers}`.")
        return

    if _NUM_WORKERS_PATH.exists():
        print_info(f"Number of workers set to `{num_workers}`.")
    else:
        print_info(f"Number of workers not set. Defaults to `{num_workers}`.")


def get_num_workers(download: bool = False) -> int:
    from helper import debug

    if debug.is_active():
        return 1

    num_workers = os.getenv("NUM_WORKERS", None)
    if num_workers is None:
        raise AssertionError(
            f"Number of workers not set. `set_num_workers` must be called in {path.helper()}/clt/main.py"
        )
    else:
        return min(int(num_workers), _MAX_NUM_WORKERS_DOWNLOAD) if download else int(num_workers)


def set_num_workers(num_workers: Optional[int] = None) -> None:
    if num_workers is not None and num_workers < 1:
        raise ValueError(f"`num_workers` ({num_workers}) must be > 0.")

    if num_workers is None:
        if _NUM_WORKERS_PATH.exists():
            num_workers_ = int(_removesuffix(file.read(path=_NUM_WORKERS_PATH, unlock=True), "\n"))
        else:
            num_workers_ = get_num_logical_cores() - 1
    else:
        num_workers_ = int(num_workers)

    os.environ["NUM_WORKERS"] = str(num_workers_)


# pu
def __set_pu(pu: Union[List[int], Literal["cpu"]]) -> None:
    try:
        pu_sanity_check(pu)
    except RuntimeError as e:
        print_error(f"Failed to set Pu. {e.args[0]}")
        sys.exit(1)

    if pu == "cpu":
        pu_ = "cpu"
        info = "CPU"
    else:
        info = _get_gpu_str(pu)
        pu_ = ",".join(map(str, pu))

    file.write(path=_PU_PATH, text=pu_, overwrite=True, lock=True)
    print_info(f"PU set to {info}.")


def __reset_pu() -> None:
    if _PU_PATH.exists():
        available_gpus = _get_available_gpus()
        default_pu_str = _get_gpu_str(available_gpus) if available_gpus else "CPU"
        _PU_PATH.unlink()
        print_info(f"PU reset. Defaults to {default_pu_str}.")
    else:
        print_error(f"Failed to reset PU. PU not set.")


def __pu_info() -> None:
    pu = get_pu()

    if pu == "cpu":
        pu_str = "CPU"
    else:
        pu_str = _get_gpu_str(pu)

    if _get_set_pu() is None:
        print_info(f"PU not set. Defaults to {pu_str}.")
    else:
        print_info(f"PU set to {pu_str}.")


def get_pu() -> Union[List[int], Literal["cpu"]]:
    pu = os.getenv("CUDA_VISIBLE_DEVICES", None)

    if pu is None:
        raise AssertionError(f"PU not set. `set_pu` must be called in {path.helper()}/clt/main.py")

    if pu == "":
        return "cpu"
    else:
        return [int(gpu_idx) for gpu_idx in pu.split(",")]


def set_pu(pu: Optional[Union[List[int], Literal["cpu"]]] = None) -> None:
    if pu is None:
        set_pu = _get_set_pu()
        available_gpus = _get_available_gpus()

        if set_pu is not None:
            pu_ = set_pu
        elif available_gpus is not None:
            pu_ = available_gpus
        else:
            pu_ = "cpu"

    else:
        pu_ = pu

    pu_sanity_check(pu_)

    os.environ["CUDA_VISIBLE_DEVICES"] = "" if pu_ == "cpu" else ",".join([str(gpu_idx) for gpu_idx in pu_])
    os.environ["MASTER_ADDR"] = _MASTER_ADDR
    os.environ["MASTER_PORT"] = str(_MASTER_PORT)


def init_pu() -> None:
    import torch

    torch.cuda.init()


def get_world_size() -> int:
    pu = get_pu()
    return 0 if pu == "cpu" else len(pu)


def get_device_str(rank: int) -> str:
    return "cpu" if get_world_size() == 0 else f"cuda:{rank}"


def is_gpu() -> bool:
    return get_world_size() > 0


def is_multi_gpu() -> bool:
    return get_world_size() > 1


def _get_available_gpus() -> Optional[List[int]]:
    import GPUtil

    n_gpus = len(GPUtil.getGPUs())
    return None if n_gpus == 0 else list(range(n_gpus))


def _get_set_pu() -> Optional[Union[List[int], Literal["cpu"]]]:
    if _PU_PATH.exists():
        set_pu = _removesuffix(file.read(path=_PU_PATH, unlock=True), "\n")
        return "cpu" if set_pu == "cpu" else [int(gpu_idx) for gpu_idx in set_pu.split(",")]
    else:
        return None


def _get_gpu_str(gpu: List[int]) -> str:
    return f"GPU(s): {', '.join([str(gpu_idx) for gpu_idx in gpu]) if gpu else 'none'}"


def pu_sanity_check(pu: Union[List[int], Literal["cpu"]]) -> None:
    if pu == "cpu":
        return

    available_gpus = _get_available_gpus()
    available_gpus_ = [] if available_gpus is None else available_gpus

    unavailable_gpus = [gpu_idx for gpu_idx in pu if gpu_idx not in available_gpus_]

    if unavailable_gpus:
        raise RuntimeError(f"Unavailable {_get_gpu_str(unavailable_gpus)}. Available {_get_gpu_str(available_gpus_)}.")


# RNG
def __set_seed(seed: int) -> None:
    from helper import debug

    if debug.is_active():
        print_error("Setting the seed is disabled in debug mode.")
        return

    file.write(path=_SEED_PATH, text=str(seed), overwrite=True, lock=True)
    print_info(f"Seed set to `{seed}`.")


def __reset_seed() -> None:
    from helper import debug

    if debug.is_active():
        print_error("Resetting the seed is disabled in debug mode.")
        return

    if _SEED_PATH.exists():
        _SEED_PATH.unlink()
        print_info(f"Seed reset. Defaults to `{_DEFAULT_SEED}`.")
    else:
        print_error(f"Failed to reset seed. Seed is not set.")


def __seed_info() -> None:
    from helper import debug

    if debug.is_active():
        print_info(f"Debugging is active. Seed defaults to `{_DEFAULT_SEED}`.")
        return

    if _SEED_PATH.exists():
        print_info(f"Seed set to `{get_seed()}`.")
    else:
        print_info(f"Seed not set. Defaults to `{_DEFAULT_SEED}`.")


def get_seed() -> int:
    from helper import debug

    if debug.is_active():
        return _DEFAULT_SEED

    seed = os.getenv("SEED", None)

    if seed is None:
        raise AssertionError(f"Seed not set. `set_seed` must be called in {path.helper()}/clt/main.py")
    else:
        return int(seed)


def set_seed(seed: Optional[int] = None) -> None:
    import numpy as np
    import torch

    if seed is None:
        if _SEED_PATH.exists():
            seed = int(_removesuffix(file.read(path=_SEED_PATH, unlock=True), "\n"))
        else:
            seed = _DEFAULT_SEED
    else:
        seed = seed
    os.environ["SEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_rng(seed: Optional[int] = None) -> random.Random:
    return random.Random(seed if seed is not None else get_seed())


# autocast
def __set_autocast(autocast: bool) -> None:
    if _AUTOCAST_PATH.exists():
        autocast_ = bool(_removesuffix(file.read(path=_AUTOCAST_PATH, unlock=True), "\n"))
        if autocast_ == autocast:
            if autocast:
                print_error("Failed to enable autocast. Autocast is already enabled.")
                sys.exit(1)
            else:
                print_error("Failed to disable autocast. Autocast is already disabled.")
                sys.exit(1)

    file.write(path=_AUTOCAST_PATH, text=str(autocast), overwrite=True, lock=True)
    print_info(f"Autocast {'enabled' if autocast else 'disabled'}.")


def __reset_autocast() -> None:
    if _AUTOCAST_PATH.exists():
        _AUTOCAST_PATH.unlink()
        default_autocast = "enabled" if _DEFAULT_AUTOCAST else "disabled"
        print_info(f"Autocast reset. Defaults to {default_autocast}.")
    else:
        print_error(f"Failed to reset autocast. Autocast is not set.")


def __autocast_info() -> None:
    if _AUTOCAST_PATH.exists():
        print_info(f"Autocast set to `{'enabled' if with_autocast() else 'disabled'}`.")
    else:
        print_info(f"Autocast not set. Defaults to {'enabled' if _DEFAULT_AUTOCAST else 'disabled'}.")


def with_autocast() -> bool:
    autocast = os.getenv("AUTOCAST", None)

    if autocast is None:
        raise AssertionError(f"Autocast not set. `set_autocast` must be called in {path.helper()}/clt/main.py")
    else:
        return bool(autocast)


# This function must be called after `set_pu`.
def set_autocast(autocast: Optional[bool] = None) -> None:
    if autocast is not None:
        autocast_ = autocast
    else:
        if _AUTOCAST_PATH.exists():
            autocast_ = bool(_removesuffix(file.read(path=_AUTOCAST_PATH, unlock=True), "\n"))
        else:
            autocast_ = False

    os.environ["AUTOCAST"] = str(autocast_)


# This function must be called after the pu is set.
def autocast_sanity_check(autocast: bool) -> None:
    import torch

    if is_gpu():
        if torch.cuda.is_bf16_supported():
            if not autocast:
                print_info("Autocast is supported but disabled.")
        else:
            if autocast:
                raise RuntimeError("Autocast is enabled but BF16 is not supported on this GPU(s).")
    else:
        if autocast:
            print_info("Autocast is enabled but PU is set to CPU. This may cause problems.")


# prefetch factor
def __set_prefetch_factor(prefetch_factor: int) -> None:
    file.write(path=_PREFETCH_FACTOR_PATH, text=str(prefetch_factor), overwrite=True, lock=True)
    print_info(f"Prefetch factor set to `{prefetch_factor}`.")


def __reset_prefetch_factor() -> None:
    if _PREFETCH_FACTOR_PATH.exists():
        _PREFETCH_FACTOR_PATH.unlink()
        print_info(f"Prefetch factor reset. Defaults to `{_DEFAULT_PREFETCH_FACTOR}`.")
    else:
        print_error(f"Failed to reset prefetch factor. Prefetch factor is not set.")


def __prefetch_factor_info() -> None:
    if _PREFETCH_FACTOR_PATH.exists():
        print_info(f"Prefetch factor set to `{get_prefetch_factor()}`.")
    else:
        print_info(f"Prefetch factor not set. Defaults to `{_DEFAULT_PREFETCH_FACTOR}`.")


def get_prefetch_factor() -> int:
    prefetch_factor = os.getenv("PREFETCH_FACTOR", None)
    if prefetch_factor is None:
        raise AssertionError(
            f"Prefetch factor not set. `set_prefetch_factor` must be called in {path.helper()}/clt/main.py"
        )
    else:
        return int(prefetch_factor)


def set_prefetch_factor(prefetch_factor: Optional[int] = None) -> None:
    if prefetch_factor is not None and prefetch_factor < 1:
        raise ValueError(f"`prefetch_factor` ({prefetch_factor}) must be >= 1.")

    if prefetch_factor is None:
        if _PREFETCH_FACTOR_PATH.exists():
            prefetch_factor_ = int(_removesuffix(file.read(path=_PREFETCH_FACTOR_PATH, unlock=True), "\n"))
        else:
            prefetch_factor_ = _DEFAULT_PREFETCH_FACTOR
    else:
        prefetch_factor_ = int(prefetch_factor)

    os.environ["PREFETCH_FACTOR"] = str(prefetch_factor_)
