import click

from helper import monitor as monitor_
from helper import system as system_
from helper.clt.custom import CustomGroup


@click.group(cls=CustomGroup, help="Manage the system.")
def system() -> None:
    pass


# info
@system.command(name="info", help="Show information about the system.")
def system_info() -> None:
    system_.__sytem_info()


# monitor
@system.group(cls=CustomGroup, help="Monitor system metrics.")
def monitor() -> None:
    pass


@monitor.command(name="cpu", help="Monitor CPU metrics.")
def monitor_cpu() -> None:
    monitor_.__cpu()


@monitor.command(name="gpu", help="Monitor GPU metrics.")
def monitor_gpu() -> None:
    monitor_.__gpu()


# num workers
@system.group(help="Manage number of workers.")
def num_workers() -> None:
    pass


@num_workers.command("set")
@click.argument("num_workers", type=click.IntRange(min=1))
def set_num_workers(num_workers: int) -> None:
    """
    Set number of workers.

    - `num_workers`: Number of workers (must be > 0).
    """
    system_.__set_num_workers(num_workers)


@num_workers.command("reset", help="Reset number of workers. Defaults to the number of physical cores minus one.")
def reset_num_workers() -> None:
    system_.__reset_num_workers()


@num_workers.command("info", help="Show information about number of workers.")
def num_workers_info() -> None:
    system_.__num_workers_info()


# pu
@system.group(help="Manage PU(s).")
def pu() -> None:
    pass


@pu.command("set")
@click.argument("pu")
def set_pu(pu: str) -> None:
    """
    Set the PU(s).

    - `pu`: A comma-separated list of GPU indices or "cpu".
    """
    pu_split = [pu_.strip() for pu_ in pu.split(",")]
    pu_split = [pu_.lower() if pu_.lower() == "cpu" else pu_ for pu_ in pu_split]

    invalid_pu = []

    for pu_ in pu_split:
        if pu_ != "cpu":
            try:
                if int(pu_) < 0:
                    invalid_pu.append(pu_)
            except ValueError:
                invalid_pu.append(pu_)

    if invalid_pu:
        raise click.UsageError(f"Invalid PU{"(S)" if len(invalid_pu) > 1 else ""}: {", ".join(invalid_pu)}.")

    if "cpu" in pu_split and len(pu_split) > 1:
        raise click.UsageError("You cannot set both CPU and GPUs.")

    system_.__set_pu("cpu" if pu_split == ["cpu"] else [int(gpu_idx) for gpu_idx in pu_split])


@pu.command("reset", help="Reset PU. Defaults to all available GPUs.")
def reset_pu() -> None:
    system_.__reset_pu()


@pu.command("info", help="Show which PU will be used.")
def pu_info() -> None:
    system_.__pu_info()


# seed
@system.group(help="Manage the random seed.")
def seed() -> None:
    pass


@seed.command("set")
@click.argument("seed", type=click.INT)
def set_seed(seed: int) -> None:
    """
    Set RNG seed.

    - `seed`: RNG seed.
    """
    system_.__set_seed(seed)


@seed.command("reset", help="Reset the random seed. Defaults to random.")
def reset_seed() -> None:
    system_.__reset_seed()


@seed.command("info", help="Show the random seed.")
def seed_info() -> None:
    system_.__seed_info()


# autocast
@system.group(help="Manage autocast.")
def autocast() -> None:
    pass


@autocast.command("enable", help="Enable autocast.")
def enable_autocast() -> None:
    system_.__set_autocast(True)


@autocast.command("disable", help="Disable autocast.")
def disable_autocast() -> None:
    system_.__set_autocast(False)


@autocast.command("info", help="Show autocast status.")
def autocast_info() -> None:
    system_.__autocast_info()


# prefetch factor
@system.group(help="Manage prefetch factor.")
def prefetch_factor() -> None:
    pass


@prefetch_factor.command("set")
@click.argument("prefetch_factor", type=click.IntRange(min=1))
def set_prefetch_factor(prefetch_factor: int) -> None:
    """
    Set the prefetch factor.

    - `prefetch_factor`: Prefetch factor (must be > 0).
    """
    system_.__set_prefetch_factor(prefetch_factor)


@prefetch_factor.command("reset", help="Reset the prefetch factor. Defaults to 1.")
def reset_prefetch_factor() -> None:
    system_.__reset_prefetch_factor()


@prefetch_factor.command("info", help="Show the prefetch factor.")
def prefetch_factor_info() -> None:
    system_.__prefetch_factor_info()
