insorderset
===========

**insorderset** provides two insertion-ordered set types for Python:

* :class:`~insorderset.OrderedSet` — mutable, like :class:`set`
* :class:`~insorderset.FrozenOrderedSet` — immutable and hashable, like :class:`frozenset`

Both types preserve the order in which elements were first inserted, while
offering the full ``set``/``frozenset`` API including binary operators,
pickle support, and ``isinstance`` compatibility with
:class:`collections.abc.Set` / :class:`collections.abc.MutableSet`.

Quick start
-----------

.. code-block:: python

   from insorderset import OrderedSet, FrozenOrderedSet

   s = OrderedSet(["b", "a", "c", "a"])
   assert list(s) == ["b", "a", "c"]   # insertion order preserved

   s.add("z")
   s.discard("b")
   assert s == {"a", "c", "z"}         # equality with built-in set

   fos = FrozenOrderedSet([1, 2, 3])
   assert hash(fos) == hash(frozenset([1, 2, 3]))  # hash-compatible
   d = {fos: "value"}                               # usable as dict key

.. toctree::
   :maxdepth: 2
   :caption: Contents

   api
