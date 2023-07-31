import time

from typing import Any, Callable, Iterable, Optional, TypeVar

T = TypeVar("T")


def tail(
    iterable: Iterable[T],
    default: T = None,
) -> T:
    _tail = default
    for _tail in iterable:
        pass
    return _tail


def time_it(
    func: Callable,
    args: Optional[list[Any]] = None,
    kwargs: Optional[dict[str, Any]] = None,
    *,
    current_time: float = 0.0,
):
    if args is None:
        args = []

    if kwargs is None:
        kwargs = {}

    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()

    return end - start + current_time, result
