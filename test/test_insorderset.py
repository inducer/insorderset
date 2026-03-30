"""
Tests for insorderset.OrderedSet and insorderset.FrozenOrderedSet.
Adapted from https://github.com/matthiasdiener/orderedsets/tree/main/test
"""

from __future__ import annotations

import pickle
from typing import AbstractSet, Any, FrozenSet, Generator, Set, Type, TypeVar, Union

import pytest

from insorderset import FrozenOrderedSet, OrderedSet

T = TypeVar("T")

set_types = (OrderedSet, FrozenOrderedSet, set, frozenset)
ordered_set_types = (OrderedSet, FrozenOrderedSet)
mutable_set_types = (OrderedSet, set)
immutable_set_types = (FrozenOrderedSet, frozenset)

T_set = Union[Type[OrderedSet], Type[FrozenOrderedSet], Type[Set], Type[FrozenSet]]

all_set_types = pytest.mark.parametrize("cls", set_types)
all_ordered_set_types = pytest.mark.parametrize("cls", ordered_set_types)
all_mutable_set_types = pytest.mark.parametrize("cls", mutable_set_types)
all_immutable_set_types = pytest.mark.parametrize("cls", immutable_set_types)


def _char_range() -> Generator[str, None, None]:
    import string
    yield from string.ascii_lowercase + string.ascii_uppercase


def test_no_PYTHONHASHSEED() -> None:  # noqa: N802
    import os
    if "PYTHONHASHSEED" in os.environ:
        val = os.environ["PYTHONHASHSEED"]
        assert val == "random", (
            "PYTHONHASHSEED must not be set (or set to 'random') for these tests"
        )


@all_set_types
def test_init(cls: T_set) -> None:
    s = cls(["d", "a", "d", "a"])
    assert len(s) == 2
    assert s == {"d", "a"}

    s = cls()
    assert s == set()

    s = cls([])
    assert s == set()
    s = cls(())
    assert s == set()
    s = cls({})
    assert s == set()

    with pytest.raises(TypeError):
        s = cls(None)  # type: ignore[call-overload]

    with pytest.raises(TypeError):
        s = cls(42)  # type: ignore[call-overload]

    # String is iterable
    s = cls("d")
    assert s == {"d"}

    with pytest.raises(TypeError):
        s = cls("a", "b")  # type: ignore[call-overload,call-arg]

    with pytest.raises(TypeError):
        s = cls([], [])  # type: ignore[call-overload,call-arg]


@all_ordered_set_types
def test_simple_ordered(cls: T_set) -> None:
    s = cls(["d", "a", "d", "a"])
    assert list(s) == ["d", "a"]


@all_set_types
def test_str_repr(cls: T_set) -> None:
    # Same logic as the original orderedsets test:
    # set uses plain {…}, everything else uses ClassName({…})
    cls_name = "" if issubclass(cls, set) else cls.__name__ + "("
    end = "" if issubclass(cls, set) else ")"

    s = cls([1])
    assert repr(s) == str(s) == cls_name + "{1}" + end

    s = cls(["d"])
    assert repr(s) == str(s) == cls_name + "{'d'}" + end

    s = cls()
    if issubclass(cls, set):
        assert repr(s) == str(s) == "set()"
    else:
        assert repr(s) == str(s) == f"{cls_name}" + end

    if cls in ordered_set_types:
        s = cls([1, 4, 1, 4])
        assert repr(s) == str(s) == cls_name + "{1, 4}" + end

        s = cls(["d", "a", "d", "a"])
        assert repr(s) == str(s) == cls_name + "{'d', 'a'}" + end

        s = cls(["a", "d", "a", "d", "a"])
        assert repr(s) == str(s) == cls_name + "{'a', 'd'}" + end

        s = cls()
        assert repr(s) == str(s) == f"{cls_name}" + end


@all_set_types
def test_len(cls: T_set) -> None:
    s = cls(["d", "a", "d", "a"])
    assert len(s) == 2


@all_set_types
def test_in(cls: T_set) -> None:
    s = cls(["d", "a", "d", "a"])
    assert "d" in s
    assert "b" not in s


@all_set_types
def test_eq(cls: T_set) -> None:
    s1 = cls(["d", "a", "d", "a"])
    s2 = cls(["d", "d", "a"])
    s3 = set(["d", "a", "d", "a"])  # noqa: C405
    assert s1 == s2
    assert s1 == s3
    assert s2 == s1
    assert s3 == s1
    assert s2 == s3 == s1

    s4 = cls(["d"])
    assert s1 != s4

    assert s1 != ["d"]
    if cls in ordered_set_types:
        assert ["d", "a"] == list(s1)

    if cls in immutable_set_types:
        assert hash(s1) == hash(s2)
        assert hash(s1) != hash(s4)


@all_set_types
def test_isdisjoint(cls: T_set) -> None:
    s1 = cls([3, 1, 2])
    s2 = cls([1, 7])
    assert not s1.isdisjoint(s2)

    s3 = cls([10, 7])
    assert s1.isdisjoint(s3)


@all_immutable_set_types
def test_remove_discard_immutable(cls: T_set) -> None:
    s = cls(["d", "a", "d", "a"])
    with pytest.raises(AttributeError):
        s.remove(17)  # type: ignore[union-attr]
    with pytest.raises(AttributeError):
        s.discard(17)  # type: ignore[union-attr]


@all_mutable_set_types
def test_remove_discard_mutable(cls: T_set) -> None:
    s = cls(["d", "a", "d", "a"])

    with pytest.raises(KeyError):
        s.remove("Y")
    s.discard("Y")

    assert s == cls(["d", "a"])

    s.remove("d")
    assert s == {"a"}
    assert isinstance(s, cls)


@all_immutable_set_types
def test_clear_immutable(cls: T_set) -> None:
    s = cls(["d", "a", "d", "a"])
    with pytest.raises(AttributeError):
        s.clear()  # type: ignore[union-attr]


@all_mutable_set_types
def test_clear_mutable(cls: T_set) -> None:
    s = cls(["d", "a", "d", "a"])
    s.clear()
    assert len(s) == 0
    assert s == cls()
    assert isinstance(s, cls)


@all_set_types
def test_convert_to_set(cls: T_set) -> None:
    assert {1, 2, 3} == set(cls([3, 1, 2]))
    assert {1, 2, 3} == frozenset(cls([3, 1, 2]))


@all_ordered_set_types
def test_tolist(cls: T_set) -> None:
    assert ["c", "a", "b"] == list(cls(["c", "a", "b"]))


@all_set_types
def test_hash(cls: T_set) -> None:
    s1 = cls(["d", "a", "d", "a"])

    if cls in mutable_set_types:
        with pytest.raises(TypeError):
            hash(s1)
    else:
        assert hash(s1)
        assert hash(s1) == hash(s1)

        s2 = cls(["d", "a", "d", "a"])
        assert s1 == s2
        assert hash(s1) == hash(s2)

        s3 = cls([4, 1, 4, 1, 5])
        assert s1 != s3
        assert hash(s1) != hash(s3)


def test_hash_value() -> None:
    fos: FrozenOrderedSet = FrozenOrderedSet([1, 2, 3])
    fs = frozenset([1, 2, 3])
    assert fs == fos
    assert hash(fs) == hash(fos)

    fos2: FrozenOrderedSet = FrozenOrderedSet(["a", "b", "c"])
    fs2 = frozenset(["a", "b", "c"])
    assert fs2 == fos2
    assert hash(fs2) == hash(fos2)


@all_immutable_set_types
def test_update_immutable(cls: T_set) -> None:
    s = cls([1, 6, 8])
    with pytest.raises(AttributeError):
        s.update([6])  # type: ignore[union-attr]


@all_mutable_set_types
def test_update_mutable(cls: T_set) -> None:
    s = cls(["a", "f", "h"])

    if cls in ordered_set_types:
        assert list(s) == ["a", "f", "h"]

    s.update(range(10, 100000))
    for v in range(10, 100000):
        s.remove(v)

    if cls in ordered_set_types:
        assert list(s) == ["a", "f", "h"]

    assert len(s) == 3
    assert s == {"a", "f", "h"}
    assert isinstance(s, cls)


@all_set_types
def test_intersection(cls: T_set) -> None:
    s1 = cls(["c", "a", "b"])
    s2 = cls(["a", "g"])
    assert cls(["a"]) == s1.intersection(s2)
    assert s1 == s1.intersection(s1)
    assert cls(["a", "b"]) == s1.intersection(["a", "b"])
    assert isinstance(s1.intersection(["b", "c"]), cls)

    if cls in ordered_set_types:
        assert list(s1.intersection(["a", "b", "c"])) == ["c", "a", "b"]
        assert list(s1.intersection(["b", "a"])) == ["a", "b"]

    assert s1.intersection() == s1
    assert s1.intersection([]) == cls()


@all_set_types
def test_intersection_multiple_args(cls: T_set) -> None:
    s1 = cls(["c", "a", "b"])
    s2 = cls(["a", "g"])
    assert cls(["a"]) == s1.intersection(s2, s2)
    assert s1 == s1.intersection(s1, s1)

    if cls in ordered_set_types:
        assert list(s1.intersection(["a", "b", "c"], ["a", "b", "c"], s1)) \
            == ["c", "a", "b"]


@all_immutable_set_types
def test_intersection_update_immutable(cls: T_set) -> None:
    s1 = cls([3, 1, 2])
    s2 = cls([1, 7])
    with pytest.raises(AttributeError):
        s1.intersection_update(s2)  # type: ignore[union-attr]


@all_mutable_set_types
def test_intersection_update_mutable(cls: T_set) -> None:
    s1 = cls(["c", "a", "b"])
    s2 = cls(["a", "g"])

    s1.intersection_update(s2)
    assert cls(["a"]) == s1

    s1.intersection_update()
    assert cls(["a"]) == s1

    s1.intersection_update([])
    assert cls() == s1

    if cls in ordered_set_types:
        s3 = cls(["c", "a", "b"])
        s3.intersection_update(["b", "a"])
        s3.intersection_update({"b", "a"})
        assert list(s3) == ["a", "b"]
        assert isinstance(s3, cls)


@all_immutable_set_types
def test_add_immutable(cls: T_set) -> None:
    s = cls([3, 1, 2])
    with pytest.raises(AttributeError):
        s.add(0)  # type: ignore[union-attr]


@all_mutable_set_types
def test_add_mutable(cls: T_set) -> None:
    s = cls(["c", "a", "b"])
    s.add("z")
    assert cls(["c", "a", "b", "z"]) == s
    if cls in ordered_set_types:
        assert list(s) == ["c", "a", "b", "z"]
    assert isinstance(s, cls)


@all_set_types
def test_copy(cls: T_set) -> None:
    s1 = cls(["c", "a", "b"])
    s2 = s1.copy()
    assert s1 == s2
    assert type(s1) is type(s2)

    if not isinstance(s1, frozenset):
        assert s1 is not s2

    if cls in ordered_set_types:
        assert list(s1) == list(s2) == ["c", "a", "b"]


@all_set_types
def test_difference(cls: T_set) -> None:
    s1 = cls(["c", "a", "b"])
    s2 = cls(["a", "g"])
    assert cls(["c", "b"]) == s1.difference(s2)
    assert s1.difference(s1) == cls()
    assert s1.difference() == s1
    assert s1.difference(["a", "b", "c"]) == cls()
    assert s1.difference(["b"], ["c"]) == cls(["a"])

    if cls in ordered_set_types:
        s3 = cls(_char_range())
        s3 = s3.difference(["a"])
        s3_res = list(_char_range())
        s3_res.remove("a")
        assert list(s3) == s3_res

        s3 = cls(_char_range())
        s3 = s3.difference(["b"], ["Y"])
        s3_res = list(_char_range())
        s3_res.remove("b")
        s3_res.remove("Y")
        assert list(s3) == s3_res


@all_set_types
def test_difference_multiple_args(cls: T_set) -> None:
    s1 = cls(["c", "a", "b"])
    s2 = cls(["a", "g"])
    assert cls(["c", "b"]) == s1.difference(s2, s2)
    assert s1.difference(s1, s1) == cls()

    if cls in ordered_set_types:
        s3 = cls(_char_range())
        s3 = s3.difference(["a"], ["b", "c"], ["Z", "Y"])
        s3_res = list(_char_range())
        s3_res.remove("a")
        s3_res.remove("b")
        s3_res.remove("c")
        s3_res.remove("Y")
        s3_res.remove("Z")
        assert list(s3) == s3_res


@all_immutable_set_types
def test_difference_update_immutable(cls: T_set) -> None:
    s1 = cls([3, 1, 2])
    s2 = cls([1, 7])
    with pytest.raises(AttributeError):
        s1.difference_update(s2)  # type: ignore[union-attr]


@all_mutable_set_types
def test_difference_update_mutable(cls: T_set) -> None:
    s1 = cls(["c", "a", "b"])
    s2 = cls(["a", "g"])

    s1.difference_update(s2)
    assert cls(["c", "b"]) == s1

    if cls in ordered_set_types:
        assert list(s1) == ["c", "b"]


@all_set_types
def test_symmetric_difference(cls: T_set) -> None:
    s1 = cls(["c", "a", "b"])
    s2 = cls(["a", "g"])
    assert cls(["c", "b", "g"]) == s1.symmetric_difference(s2)

    if cls in ordered_set_types:
        assert list(s1.symmetric_difference(s2)) == ["c", "b", "g"]


@all_immutable_set_types
def test_symmetric_difference_update_immutable(cls: T_set) -> None:
    s1 = cls([3, 1, 2])
    s2 = cls([1, 7])
    with pytest.raises(AttributeError):
        s1.symmetric_difference_update(s2)  # type: ignore[union-attr]


@all_mutable_set_types
def test_symmetric_difference_update(cls: T_set) -> None:
    s1 = cls(["c", "a", "b"])
    s2 = cls(["a", "g"])

    s1.symmetric_difference_update(s2)
    assert cls(["c", "b", "g"]) == s1

    if cls in ordered_set_types:
        assert list(s1) == ["c", "b", "g"]


@all_set_types
def test_union(cls: T_set) -> None:
    s1 = cls(["c", "a", "b"])
    s2 = cls(["a", "g"])
    s3 = cls(["Z"])
    assert cls(["c", "a", "b", "g"]) == s1.union(s2)
    assert cls(["c", "a", "b", "Z"]) == s1.union(s3)

    if cls in ordered_set_types:
        assert list(s1.union(s2)) == ["c", "a", "b", "g"]
        assert list(s1.union(s3)) == ["c", "a", "b", "Z"]


@all_set_types
def test_union_multiple_args(cls: T_set) -> None:
    s1 = cls(["c", "a", "b"])
    s2 = cls(["a", "g"])
    s3 = cls(["Z"])
    assert cls(["c", "a", "b", "g", "Z"]) == s1.union(s2, s3)

    if cls in ordered_set_types:
        assert list(s1.union(s2, s3)) == ["c", "a", "b", "g", "Z"]


@all_set_types
def test_op_and(cls: T_set) -> None:
    s1 = cls(["c", "a", "b"])
    s2 = cls(["a", "g", "b"])
    s3 = cls(["Z", "b", "c"])
    assert cls(["a", "b"]) == s1 & s2
    assert cls(["c", "b"]) == s1 & s3

    assert s1 & s1 == s1
    assert s1 & cls(["Z"]) == cls([])

    if cls in ordered_set_types:
        assert list(s1 & s2) == ["a", "b"]
        assert list(s1 & s3) == ["c", "b"]


@all_set_types
def test_op_iand(cls: T_set) -> None:
    s1 = cls(["c", "a", "b"])
    s2 = cls(["a", "g", "b"])
    s3 = cls(["Z", "b", "c"])

    s1 &= s2
    assert cls(["a", "b"]) == s1

    if cls in ordered_set_types:
        assert list(s1) == ["a", "b"]

    s1 &= s3
    assert cls(["b"]) == s1


@all_set_types
def test_op_or(cls: T_set) -> None:
    s1 = cls(["c", "a", "b"])
    s2 = cls(["a", "g"])
    s3 = cls(["Z"])

    assert cls(["c", "a", "b", "g"]) == s1 | s2
    assert cls(["c", "a", "b", "Z"]) == s1 | s3

    if cls in ordered_set_types:
        assert list(s1 | s2) == ["c", "a", "b", "g"]
        assert list(s1 | s3) == ["c", "a", "b", "Z"]

    assert s1 | s1 == s1


@all_set_types
def test_op_ior(cls: T_set) -> None:
    s1 = cls(["c", "a", "b"])
    s2 = cls(["a", "g"])
    s3 = cls(["Z"])

    s1 |= s2
    assert cls(["c", "a", "b", "g"]) == s1

    if cls in ordered_set_types:
        assert ["c", "a", "b", "g"] == list(s1)

    s1 |= s3
    assert cls(["c", "a", "b", "g", "Z"]) == s1

    if cls in ordered_set_types:
        assert ["c", "a", "b", "g", "Z"] == list(s1)


@all_set_types
def test_op_sub(cls: T_set) -> None:
    s1 = cls(["c", "a", "b"])
    s2 = cls(["a", "g"])
    s3 = cls(["Z"])

    assert cls(["c", "b"]) == s1 - s2
    assert cls(["c", "a", "b"]) == s1 - s3

    if cls in ordered_set_types:
        assert list(s1 - s2) == ["c", "b"]
        assert list(s1 - s3) == ["c", "a", "b"]


@all_set_types
def test_op_isub(cls: T_set) -> None:
    s1 = cls(["c", "a", "b"])
    s2 = cls(["a", "g"])
    s3 = cls(["Z"])

    s1 -= s2
    assert cls(["c", "b"]) == s1

    if cls in ordered_set_types:
        assert ["c", "b"] == list(s1)
    s1 -= s3
    assert cls(["c", "b"]) == s1


@all_set_types
def test_op_xor(cls: T_set) -> None:
    s1 = cls(["c", "a", "b"])
    s2 = cls(["a", "g"])
    s3 = cls(["Z"])

    assert cls(["c", "b", "g"]) == s1 ^ s2
    assert cls(["c", "a", "b", "Z"]) == s1 ^ s3

    if cls in ordered_set_types:
        assert list(s1 ^ s2) == ["c", "b", "g"]
        assert list(s1 ^ s3) == ["c", "a", "b", "Z"]


@all_set_types
def test_op_ixor(cls: T_set) -> None:
    s1 = cls(["c", "a", "b"])
    s2 = cls(["a", "g"])
    s3 = cls(["Z"])
    s1 ^= s2

    assert cls(["c", "b", "g"]) == s1
    if cls in ordered_set_types:
        assert list(s1) == ["c", "b", "g"]
    s1 ^= s3
    assert cls(["c", "b", "Z", "g"]) == s1


@all_set_types
def test_issubset(cls: T_set) -> None:
    s1 = cls([3, 1, 2])
    s2 = cls([1, 7])
    assert not s2.issubset(s1)
    assert cls().issubset(s1)

    if cls in mutable_set_types:
        s2.discard(7)  # type: ignore[union-attr]
        assert s2.issubset(s1)


@all_set_types
def test_issuperset(cls: T_set) -> None:
    s1 = cls([3, 1, 2])
    s2 = cls([1, 7])
    assert not s1.issuperset(s2)

    if cls in mutable_set_types:
        s2.discard(7)  # type: ignore[union-attr]
        assert s1.issuperset(s2)
        assert cls(s1).issuperset(s2)


@all_immutable_set_types
def test_pop_immutable(cls: T_set) -> None:
    s1 = cls([3, 1, 2])
    with pytest.raises(AttributeError):
        s1.pop()  # type: ignore[union-attr]


@all_mutable_set_types
def test_pop_mutable(cls: T_set) -> None:
    s1 = cls(["c", "a", "b"])

    p = s1.pop()
    assert len(s1) == 2

    if cls in ordered_set_types:
        assert p == "b"
        assert list(s1) == ["c", "a"]


@all_set_types
def test_op_le(cls: T_set) -> None:
    s1 = cls([3, 1, 2])
    s2 = cls([1, 7])

    assert not (s2 <= s1)
    s3 = cls([1, 2])
    assert s3 <= s1
    assert not s2.issubset(s1)


@all_set_types
def test_op_lt(cls: T_set) -> None:
    s1 = cls([3, 1, 2])
    s2 = cls([1, 7])
    assert not (s2 < s1)
    s3 = cls([1, 2])
    assert s3 < s1
    assert not s2.issubset(s1)


@all_set_types
def test_op_ge(cls: T_set) -> None:
    s1 = cls([3, 1, 2])
    s2 = cls([1, 7])
    assert not (s2 >= s1)
    s3 = cls([1, 2])
    assert s1 >= s3
    assert not s2.issubset(s1)


@all_set_types
def test_op_gt(cls: T_set) -> None:
    s1 = cls([3, 1, 2])
    s2 = cls([1, 7])
    assert not (s2 > s1)
    s3 = cls([1, 2])
    assert s1 > s3
    assert not s2.issubset(s1)


@all_set_types
def test_bool(cls: T_set) -> None:
    assert not cls()
    assert cls([1])
    assert not bool(cls())
    assert bool(cls([1]))


@all_set_types
def test_ordering(cls: T_set) -> None:
    lst = list(range(10))

    oset1 = cls(lst)
    oset2 = cls(lst)

    assert oset2 <= oset1
    assert oset2 <= set(oset1)
    assert oset1 >= oset2
    assert oset1 >= set(oset2)

    oset3 = cls(lst[:-1])

    assert oset3 < oset1
    assert oset3 < set(oset1)

    assert oset1 > oset3
    assert oset1 > set(oset3)

    oset4 = cls(lst[1:])

    assert not (oset3 < oset4)
    assert not (oset3 < set(oset4))
    assert not (oset3 >= oset4)
    assert not (oset3 >= set(oset4))


@all_ordered_set_types
def test_isinstance(cls: T_set) -> None:
    from collections.abc import Set as abc_Set
    assert isinstance(cls(), AbstractSet)
    assert isinstance(cls(), abc_Set)
    assert not isinstance(cls(), set)
    assert not isinstance(cls(), frozenset)

    if cls in mutable_set_types:
        assert isinstance(cls(), OrderedSet)
        assert not isinstance(cls(), FrozenOrderedSet)
    else:
        assert isinstance(cls(), FrozenOrderedSet)
        assert not isinstance(cls(), OrderedSet)


def test_large_set() -> None:
    """Stress test: insert and delete many elements."""
    n = 10000
    s = OrderedSet(range(n))
    assert len(s) == n
    assert list(s) == list(range(n))

    for i in range(0, n, 2):
        s.discard(i)
    assert len(s) == n // 2
    assert list(s) == list(range(1, n, 2))

    s2 = FrozenOrderedSet(range(n))
    assert len(s2) == n


@all_set_types
def test_pickle(cls: T_set) -> None:
    s = cls(["c", "a", "b"])
    for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
        data = pickle.dumps(s, protocol)
        s2 = pickle.loads(data)
        assert s == s2
        assert type(s) is type(s2)
        if cls in ordered_set_types:
            assert list(s) == list(s2)


def test_frozen_as_dict_key() -> None:
    fos = FrozenOrderedSet([1, 2, 3])
    d = {fos: "value"}
    assert d[fos] == "value"
    assert d[FrozenOrderedSet([1, 2, 3])] == "value"


def test_frozen_in_set() -> None:
    fos1 = FrozenOrderedSet([1, 2, 3])
    fos2 = FrozenOrderedSet([4, 5, 6])
    s = {fos1, fos2}
    assert len(s) == 2
    assert fos1 in s


def test_duplicate_insertion() -> None:
    s = OrderedSet()
    for _ in range(100):
        s.add("x")
    assert len(s) == 1
    assert list(s) == ["x"]


def test_iteration_after_deletions() -> None:
    s = OrderedSet(range(10))
    for i in range(5):
        s.discard(i)
    assert list(s) == list(range(5, 10))


def test_hash_cache() -> None:
    """FrozenOrderedSet hash is stable and cached."""
    fos = FrozenOrderedSet([1, 2, 3])
    h1 = hash(fos)
    h2 = hash(fos)
    assert h1 == h2


def test_empty_operations() -> None:
    s = OrderedSet()
    assert s.difference(s) == OrderedSet()
    assert s.intersection(s) == OrderedSet()
    assert s.union(s) == OrderedSet()
    assert s.symmetric_difference(s) == OrderedSet()


def test_copy_independence() -> None:
    s1 = OrderedSet([1, 2, 3])
    s2 = s1.copy()
    s2.add(4)
    assert 4 not in s1
    assert 4 in s2
