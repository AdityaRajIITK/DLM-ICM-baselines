"""
Minimal local subset of the external `rfutils` package.

This repository originally depended on `rfutils` from GitHub, but the runtime
environment may not allow outgoing network fetches. We only implement the
small set of helpers actually imported by this codebase.
"""

from __future__ import annotations

import itertools as _it
from collections import deque as _deque
from typing import Callable, Iterable, Iterator, Tuple, TypeVar

T = TypeVar("T")
U = TypeVar("U")


def the_only(xs: Iterable[T]) -> T:
    """Return the single element from `xs`, erroring if it isn't singleton."""
    lst = list(xs)
    if len(lst) != 1:
        raise ValueError(f"Expected exactly one element, got {len(lst)}")
    return lst[0]


def partition(pred: Callable[[T], bool], xs: Iterable[T]) -> Tuple[Iterator[T], Iterator[T]]:
    """
    Split `xs` into two iterators: (items where pred(x) is True, pred(x) is False).

    The input is fully materialized to avoid consuming an iterator twice.
    """
    items = list(xs)
    a = [x for x in items if pred(x)]
    b = [x for x in items if not pred(x)]
    return iter(a), iter(b)


def flatmap(f: Callable[[T], Iterable[U]], xs: Iterable[T]) -> Iterator[U]:
    """Map each x in xs to an iterable and flatten by one level."""
    return _it.chain.from_iterable(map(f, xs))


def sliding(xs: Iterable[T], n: int) -> Iterator[Tuple[T, ...]]:
    """Yield length-n sliding windows over xs."""
    if n <= 0:
        raise ValueError("n must be >= 1")
    it = iter(xs)
    window: _deque[T] = _deque(_it.islice(it, n), maxlen=n)
    if len(window) < n:
        return
        yield  # pragma: no cover
    yield tuple(window)
    for x in it:
        window.append(x)
        yield tuple(window)

