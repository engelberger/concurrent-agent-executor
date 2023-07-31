from typing import Iterable, TypeVar

T = TypeVar("T")


def tail(
    iterable: Iterable[T],
    default: T = None,
) -> T:
    _tail = default
    for _tail in iterable:
        pass
    return _tail
