"""
Minimal subset of `rfutils.nondet` used by this repository.

`nondet_map` is used to enumerate nondeterministic choices (cartesian product)
across multiple arguments.
"""

from __future__ import annotations

import itertools as it
from typing import Callable, Iterable, Iterator, Sequence, TypeVar

T = TypeVar("T")
U = TypeVar("U")


def nondet_map(f: Callable[[T], Iterable[U]], xs: Sequence[T]) -> Iterator[tuple[U, ...]]:
    """
    Given a function `f` that maps each input to an iterable of possibilities,
    yield all cartesian combinations of those possibilities.
    """
    return it.product(*(f(x) for x in xs))

