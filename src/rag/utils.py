from __future__ import annotations

from itertools import islice
from typing import Iterable, Iterator, TypeVar

T = TypeVar("T")


def batched(iterable: Iterable[T], size: int) -> Iterator[list[T]]:
    """Yield lists of up to `size` items (itertools.batched needs Python 3.12)."""
    if size < 1:
        raise ValueError("size must be >= 1")
    it = iter(iterable)
    while batch := list(islice(it, size)):
        yield batch
