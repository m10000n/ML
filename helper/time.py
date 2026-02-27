import datetime

from helper import helper_

##### config start #####
_FORMAT = "%Y-%m-%d %H:%M:%S"
##### config end #####


def now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def now_str() -> str:
    return f"{now().strftime(_FORMAT)} (UTC)"


def h_min_sec_str(delta: datetime.timedelta, truncate: bool = False) -> str:
    total_seconds = helper_.round_to_int(x=delta.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    result = ""
    if truncate:
        if hours > 0:
            result += f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            result += f"{minutes}m {seconds}s"
        elif seconds > 0:
            result += f"{seconds}s"
    else:
        result = f"{hours}h {minutes}m {seconds}s"

    return result
