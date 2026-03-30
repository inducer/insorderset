"""Insertion-ordered set for Python.

Provides :class:`OrderedSet` (mutable) and :class:`FrozenOrderedSet`
(immutable, hashable) — drop-in replacements for :class:`set` and
:class:`frozenset` that preserve insertion order.
"""

from __future__ import annotations

from collections.abc import MutableSet, Set

from insorderset._insorderset_core import FrozenOrderedSet, OrderedSet

# Register with the Abstract Base Classes so that isinstance checks work:
#   isinstance(OrderedSet(), collections.abc.Set)   -> True
#   isinstance(OrderedSet(), typing.AbstractSet)     -> True (same ABC)
Set.register(OrderedSet)
Set.register(FrozenOrderedSet)
MutableSet.register(OrderedSet)

__all__ = ["OrderedSet", "FrozenOrderedSet"]
