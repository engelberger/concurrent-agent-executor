import time
from typing import Any, Callable, Optional


def time_it(
    func: Callable,
    *,
    args: Optional[list[Any]],
    kwargs: Optional[dict[Any]],
    current_time: float = 0,
) -> tuple[float, Any]:
    if args is None:
        args = []

    if kwargs is None:
        kwargs = {}

    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()

    return end - start + current_time, result
