from decimal import ROUND_HALF_UP, Decimal
from typing import Any


def round_to_str(x: float, digits: int) -> str:
    q = Decimal(1).scaleb(-digits)
    d = Decimal(str(x)).quantize(q, rounding=ROUND_HALF_UP)
    return f"{d:.{digits}f}"


def round_to_int(x: float) -> int:
    return int(round_to_str(x=x, digits=0))


def flatten(x: list[Any]) -> list[Any]:
    result = []

    def _rec_flatten(x: Any) -> None:
        if isinstance(x, (list, tuple)):
            for item in x:
                _rec_flatten(item)
        else:
            result.append(x)

    _rec_flatten(x)
    return result
