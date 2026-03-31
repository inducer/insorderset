"""Insertion-ordered set for Python.

Provides :class:`OrderedSet` (mutable) and :class:`FrozenOrderedSet`
(immutable, hashable) — drop-in replacements for :class:`set` and
:class:`frozenset` that preserve insertion order.
"""

from __future__ import annotations

from collections.abc import MutableSet as abc_MutableSet
from collections.abc import Set as abc_Set

from insorderset._insorderset_core import FrozenOrderedSet, OrderedSet

abc_Set.register(OrderedSet)  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
abc_Set.register(FrozenOrderedSet)  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
abc_MutableSet.register(OrderedSet)  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]

__all__ = ["FrozenOrderedSet", "OrderedSet"]
