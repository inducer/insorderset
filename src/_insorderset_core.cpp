/*
 * _insorderset_core.cpp
 *
 * Python C extension (via nanobind) implementing insertion-ordered sets.
 * Core data structure inspired by CPython's Objects/dictobject.c —
 * a compact hash table (indices array + dense entries array) but without
 * the "value" half of the key/value pair.
 *
 * Exposed types:
 *   OrderedSet       – mutable, unhashable
 *   FrozenOrderedSet – immutable, hashable (same hash as frozenset)
 *
 * Parts derived from
 * https://github.com/python/cpython/blob/62a6e898e017c9878490544f6a227b8a187a949c/Objects/dictobject.c
 * Copyright/license as specified in
 * https://github.com/python/cpython/blob/62a6e898e017c9878490544f6a227b8a187a949c/LICENSE
 */

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <Python.h>

#include <cassert>
#include <optional>
#include <sstream>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

/* =========================================================================
   InsOrderedSetImpl – the underlying compact hash table
   =========================================================================

   Layout (CPython dict-compact approach, keys-only variant):

     indices[slot]  in { IX_EMPTY(-1), IX_DUMMY(-2), or index >=0 into entries }
     hashes[i]      – cached Py_hash_t for entry i
     keys[i]        – owning PyObject* for entry i  (nullptr when deleted)
     alive[i]       – true while entry i is active

   Probing: slot = hash & mask
            loop: perturb >>= 5; slot = (5*slot + perturb + 1) & mask
   Resize trigger: fill > 2/3 * capacity
   After resize the entries array is compacted (deleted slots removed).
*/

struct InsOrderedSetImpl {
    static constexpr Py_ssize_t IX_EMPTY = -1;
    static constexpr Py_ssize_t IX_DUMMY = -2;

    std::vector<Py_ssize_t> indices;  // hash table
    std::vector<Py_hash_t>  hashes;   // per-entry hash
    std::vector<PyObject *> keys;     // per-entry key (owned reference)
    std::vector<bool>       alive;    // per-entry liveness flag
    Py_ssize_t              used;     // number of active entries
    Py_ssize_t              fill;     // active + dummy slots in indices

    /* ---- construction / copy / destruction ---- */

    InsOrderedSetImpl() : used(0), fill(0) {
        indices.assign(8, IX_EMPTY);
    }

    InsOrderedSetImpl(const InsOrderedSetImpl &o)
        : indices(o.indices), hashes(o.hashes),
          alive(o.alive), used(o.used), fill(o.fill)
    {
        keys.resize(o.keys.size(), nullptr);
        for (size_t i = 0; i < o.keys.size(); ++i) {
            keys[i] = o.keys[i];
            Py_XINCREF(keys[i]);
        }
    }

    InsOrderedSetImpl &operator=(const InsOrderedSetImpl &o) {
        if (this == &o) return *this;
        _release();
        indices = o.indices;
        hashes  = o.hashes;
        alive   = o.alive;
        used    = o.used;
        fill    = o.fill;
        keys.resize(o.keys.size(), nullptr);
        for (size_t i = 0; i < o.keys.size(); ++i) {
            keys[i] = o.keys[i];
            Py_XINCREF(keys[i]);
        }
        return *this;
    }

    ~InsOrderedSetImpl() { _release(); }

    void _release() {
        for (auto *k : keys) Py_XDECREF(k);
    }

    /* ---- helpers ---- */

    size_t mask() const { return indices.size() - 1; }

    /* Find index into keys[] for (hash, key); returns -1 if absent. */
    Py_ssize_t _lookup(Py_hash_t hash, PyObject *key) const {
        size_t perturb = static_cast<size_t>(hash);
        size_t slot    = static_cast<size_t>(hash) & mask();
        for (;;) {
            Py_ssize_t ix = indices[slot];
            if (ix == IX_EMPTY) return -1;
            if (ix >= 0 && alive[static_cast<size_t>(ix)] &&
                hashes[static_cast<size_t>(ix)] == hash)
            {
                int eq = PyObject_RichCompareBool(key, keys[static_cast<size_t>(ix)], Py_EQ);
                if (eq < 0) throw nb::python_error();
                if (eq) return ix;
            }
            perturb >>= 5;
            slot = (5 * slot + perturb + 1) & mask();
        }
    }

    /* ---- public operations ---- */

    bool contains(PyObject *key) const {
        Py_hash_t h = PyObject_Hash(key);
        if (h == -1) throw nb::python_error();
        return _lookup(h, key) >= 0;
    }

    /* Insert key.  Returns true if newly inserted, false if duplicate. */
    bool insert(PyObject *key) {
        Py_hash_t h = PyObject_Hash(key);
        if (h == -1) throw nb::python_error();

        size_t    perturb     = static_cast<size_t>(h);
        size_t    slot        = static_cast<size_t>(h) & mask();
        Py_ssize_t first_dummy = -1;

        for (;;) {
            Py_ssize_t ix = indices[slot];
            if (ix == IX_EMPTY) break;
            if (ix == IX_DUMMY) {
                if (first_dummy < 0) first_dummy = static_cast<Py_ssize_t>(slot);
            } else if (alive[static_cast<size_t>(ix)] &&
                       hashes[static_cast<size_t>(ix)] == h)
            {
                int eq = PyObject_RichCompareBool(key, keys[static_cast<size_t>(ix)], Py_EQ);
                if (eq < 0) throw nb::python_error();
                if (eq) return false; // already present
            }
            perturb >>= 5;
            slot = (5 * slot + perturb + 1) & mask();
        }

        /* Use the earliest dummy slot encountered, otherwise the empty slot. */
        size_t ins_slot = (first_dummy >= 0) ? static_cast<size_t>(first_dummy) : slot;
        bool   new_slot = (indices[ins_slot] == IX_EMPTY);
        auto   new_ix   = static_cast<Py_ssize_t>(keys.size());

        indices[ins_slot] = new_ix;
        if (new_slot) ++fill;

        Py_INCREF(key);
        hashes.push_back(h);
        keys.push_back(key);
        alive.push_back(true);
        ++used;

        /* Resize when fill > 2/3 capacity. */
        if (fill * 3 >= static_cast<Py_ssize_t>(indices.size()) * 2) {
            size_t new_sz = indices.size();
            /* Compact if the deletion rate is high; otherwise grow 4x. */
            if (static_cast<size_t>(used) * 4 <= fill * 3)
                new_sz = std::max(new_sz, static_cast<size_t>(used) * 2);
            else
                new_sz = indices.size() * 4;
            /* Ensure it's a power-of-2 and large enough. */
            while (new_sz < static_cast<size_t>(used) * 2) new_sz *= 2;
            _resize(new_sz);
        }
        return true;
    }

    /* Discard key.  Returns true if removed, false if absent. */
    bool discard_key(PyObject *key) {
        Py_hash_t h = PyObject_Hash(key);
        if (h == -1) throw nb::python_error();

        size_t perturb = static_cast<size_t>(h);
        size_t slot    = static_cast<size_t>(h) & mask();
        for (;;) {
            Py_ssize_t ix = indices[slot];
            if (ix == IX_EMPTY) return false;
            if (ix >= 0 && alive[static_cast<size_t>(ix)] &&
                hashes[static_cast<size_t>(ix)] == h)
            {
                int eq = PyObject_RichCompareBool(key, keys[static_cast<size_t>(ix)], Py_EQ);
                if (eq < 0) throw nb::python_error();
                if (eq) {
                    indices[slot] = IX_DUMMY;
                    alive[static_cast<size_t>(ix)] = false;
                    Py_DECREF(keys[static_cast<size_t>(ix)]);
                    keys[static_cast<size_t>(ix)] = nullptr;
                    --used;
                    return true;
                }
            }
            perturb >>= 5;
            slot = (5 * slot + perturb + 1) & mask();
        }
    }

    /* Remove key; raises KeyError if absent. */
    void remove_key(PyObject *key) {
        if (!discard_key(key)) {
            PyErr_SetObject(PyExc_KeyError, key);
            throw nb::python_error();
        }
    }

    /* Remove and return the last-inserted active element (new reference). */
    PyObject *pop_last() {
        if (used == 0) {
            PyErr_SetString(PyExc_KeyError, "pop from an empty set");
            throw nb::python_error();
        }
        for (Py_ssize_t i = static_cast<Py_ssize_t>(keys.size()) - 1; i >= 0; --i) {
            if (!alive[static_cast<size_t>(i)]) continue;

            /* Locate and mark the slot as dummy. */
            Py_hash_t h      = hashes[static_cast<size_t>(i)];
            size_t    perturb = static_cast<size_t>(h);
            size_t    slot    = static_cast<size_t>(h) & mask();
            for (;;) {
                if (indices[slot] == i) { indices[slot] = IX_DUMMY; break; }
                perturb >>= 5;
                slot = (5 * slot + perturb + 1) & mask();
            }

            PyObject *result                       = keys[static_cast<size_t>(i)];
            alive[static_cast<size_t>(i)]          = false;
            keys[static_cast<size_t>(i)]           = nullptr;
            --used;
            return result; // caller owns the reference
        }
        PyErr_SetString(PyExc_KeyError, "pop from an empty set");
        throw nb::python_error();
    }

    void clear_all() {
        _release();
        indices.assign(8, IX_EMPTY);
        hashes.clear();
        keys.clear();
        alive.clear();
        used = 0;
        fill = 0;
    }

    /* Compact + rebuild the hash table at the given size. */
    void _resize(size_t new_sz) {
        std::vector<Py_ssize_t> ni(new_sz, IX_EMPTY);
        std::vector<Py_hash_t>  nh;
        std::vector<PyObject *> nk;
        std::vector<bool>       na;
        nh.reserve(static_cast<size_t>(used));
        nk.reserve(static_cast<size_t>(used));
        na.reserve(static_cast<size_t>(used));

        size_t nmask = new_sz - 1;
        for (size_t i = 0; i < keys.size(); ++i) {
            if (!alive[i]) continue;
            Py_hash_t  h       = hashes[i];
            auto       new_ix  = static_cast<Py_ssize_t>(nk.size());
            size_t     perturb = static_cast<size_t>(h);
            size_t     slot    = static_cast<size_t>(h) & nmask;
            while (ni[slot] != IX_EMPTY) {
                perturb >>= 5;
                slot = (5 * slot + perturb + 1) & nmask;
            }
            ni[slot] = new_ix;
            nh.push_back(h);
            nk.push_back(keys[i]); // ownership transferred
            na.push_back(true);
        }
        indices = std::move(ni);
        hashes  = std::move(nh);
        keys    = std::move(nk);
        alive   = std::move(na);
        fill    = used;
    }

    /* Insert all items from a Python iterable. */
    void extend_from(PyObject *iterable) {
        PyObject *iter = PyObject_GetIter(iterable);
        if (!iter) throw nb::python_error();
        PyObject *item;
        while ((item = PyIter_Next(iter))) {
            bool ok = true;
            try { insert(item); }
            catch (...) { ok = false; }
            Py_DECREF(item);
            if (!ok) { Py_DECREF(iter); throw; }
        }
        Py_DECREF(iter);
        if (PyErr_Occurred()) throw nb::python_error();
    }
};

/* =========================================================================
   Iterator snapshot (safe against mutation during iteration)
   =========================================================================*/

struct InsOrderedSetIterator {
    std::vector<nb::object> items;
    size_t pos = 0;

    explicit InsOrderedSetIterator(const InsOrderedSetImpl &impl) {
        items.reserve(static_cast<size_t>(impl.used));
        for (size_t i = 0; i < impl.keys.size(); ++i)
            if (impl.alive[i])
                items.emplace_back(nb::borrow(impl.keys[i]));
    }

    nb::object __next__() {
        if (pos >= items.size()) throw nb::stop_iteration();
        return items[pos++];
    }

    InsOrderedSetIterator *__iter__() { return this; }
};

/* =========================================================================
   Python-exposed structs
   =========================================================================*/

struct OrderedSet {
    InsOrderedSetImpl impl;
};

struct FrozenOrderedSet {
    InsOrderedSetImpl      impl;
    mutable std::optional<Py_hash_t> cached_hash;
};

/* =========================================================================
   Helper: is the PyObject* one of our ordered-set types?
   Forward declarations needed because FrozenOrderedSet is used in OrderedSet
   methods and vice-versa.
   =========================================================================*/

static bool is_set_compatible(PyObject *obj) {
    /* Accept: set, frozenset, and anything registered as Set
       (we check via duck-typing: has __len__ and __contains__).
       We also accept numpy scalar-like objects that implement __hash__. */
    if (PyAnySet_Check(obj)) return true;
    /* For our own types we'll handle by direct isinstance checks in nb. */
    return false;
}

/* =========================================================================
   Shared implementation helpers (templated on result type)
   =========================================================================*/

/* Equality check: self.impl == other_obj (any set-like). */
template <typename Self>
static nb::object impl_eq(const Self &self, nb::handle other) {
    /* Accept built-in sets and our own types. */
    bool other_is_set = PyAnySet_Check(other.ptr());
    bool other_is_ours = nb::isinstance<OrderedSet>(other) ||
                         nb::isinstance<FrozenOrderedSet>(other);
    if (!other_is_set && !other_is_ours) return nb::not_implemented();

    const InsOrderedSetImpl *op = nullptr;
    InsOrderedSetImpl tmp_dummy; // never actually used
    if (nb::isinstance<OrderedSet>(other))
        op = &nb::cast<const OrderedSet &>(other).impl;
    else if (nb::isinstance<FrozenOrderedSet>(other))
        op = &nb::cast<const FrozenOrderedSet &>(other).impl;

    if (op) {
        if (self.impl.used != op->used) return nb::bool_(false);
        for (size_t i = 0; i < self.impl.keys.size(); ++i) {
            if (!self.impl.alive[i]) continue;
            if (op->_lookup(self.impl.hashes[i], self.impl.keys[i]) < 0)
                return nb::bool_(false);
        }
        return nb::bool_(true);
    }

    /* Built-in set / frozenset */
    if (PySet_GET_SIZE(other.ptr()) != self.impl.used) return nb::bool_(false);
    for (size_t i = 0; i < self.impl.keys.size(); ++i) {
        if (!self.impl.alive[i]) continue;
        int found = PySet_Contains(other.ptr(), self.impl.keys[i]);
        if (found < 0) throw nb::python_error();
        if (!found) return nb::bool_(false);
    }
    return nb::bool_(true);
}

/* Check whether every element of self is in `other` (any iterable/set). */
template <typename Self>
static bool impl_issubset(const Self &self, nb::handle other) {
    for (size_t i = 0; i < self.impl.keys.size(); ++i) {
        if (!self.impl.alive[i]) continue;
        int found = PySequence_Contains(other.ptr(), self.impl.keys[i]);
        if (found < 0) throw nb::python_error();
        if (!found) return false;
    }
    return true;
}

/* Check whether every element of `other` is in self. */
template <typename Self>
static bool impl_issuperset(const Self &self, nb::handle other) {
    /* Iterate other and check containment in self. */
    PyObject *iter = PyObject_GetIter(other.ptr());
    if (!iter) throw nb::python_error();
    PyObject *item;
    while ((item = PyIter_Next(iter))) {
        bool in = self.impl.contains(item);
        Py_DECREF(item);
        if (!in) { Py_DECREF(iter); return false; }
    }
    Py_DECREF(iter);
    if (PyErr_Occurred()) throw nb::python_error();
    return true;
}

/* isdisjoint */
template <typename Self>
static bool impl_isdisjoint(const Self &self, nb::handle other) {
    PyObject *iter = PyObject_GetIter(other.ptr());
    if (!iter) throw nb::python_error();
    PyObject *item;
    while ((item = PyIter_Next(iter))) {
        bool in = self.impl.contains(item);
        Py_DECREF(item);
        if (in) { Py_DECREF(iter); return false; }
    }
    Py_DECREF(iter);
    if (PyErr_Occurred()) throw nb::python_error();
    return true;
}

/* Build an InsOrderedSetImpl representing difference(self, *others). */
static InsOrderedSetImpl compute_difference(const InsOrderedSetImpl &src,
                                            nb::args others) {
    InsOrderedSetImpl result = src;
    for (size_t a = 0; a < others.size(); ++a) {
        PyObject *iter = PyObject_GetIter(others[a].ptr());
        if (!iter) throw nb::python_error();
        PyObject *item;
        while ((item = PyIter_Next(iter))) {
            bool ok = true;
            try { result.discard_key(item); }
            catch (...) { ok = false; }
            Py_DECREF(item);
            if (!ok) { Py_DECREF(iter); throw; }
        }
        Py_DECREF(iter);
        if (PyErr_Occurred()) throw nb::python_error();
    }
    return result;
}

/* Build an InsOrderedSetImpl representing intersection(self, *others). */
static InsOrderedSetImpl compute_intersection(const InsOrderedSetImpl &src,
                                              nb::args others) {
    if (others.size() == 0) return src;

    InsOrderedSetImpl result;
    for (size_t i = 0; i < src.keys.size(); ++i) {
        if (!src.alive[i]) continue;
        bool in_all = true;
        for (size_t a = 0; a < others.size() && in_all; ++a) {
            int found = PySequence_Contains(others[a].ptr(), src.keys[i]);
            if (found < 0) throw nb::python_error();
            if (!found) in_all = false;
        }
        if (in_all) result.insert(src.keys[i]);
    }
    return result;
}

/* Build an InsOrderedSetImpl representing union(self, *others). */
static InsOrderedSetImpl compute_union(const InsOrderedSetImpl &src,
                                       nb::args others) {
    InsOrderedSetImpl result = src;
    for (size_t a = 0; a < others.size(); ++a)
        result.extend_from(others[a].ptr());
    return result;
}

/* Build an InsOrderedSetImpl representing symmetric_difference(self, other). */
static InsOrderedSetImpl compute_symmdiff(const InsOrderedSetImpl &src,
                                          nb::handle other) {
    InsOrderedSetImpl result;
    /* Elements in self but not other */
    for (size_t i = 0; i < src.keys.size(); ++i) {
        if (!src.alive[i]) continue;
        int found = PySequence_Contains(other.ptr(), src.keys[i]);
        if (found < 0) throw nb::python_error();
        if (!found) result.insert(src.keys[i]);
    }
    /* Elements in other but not self */
    PyObject *iter = PyObject_GetIter(other.ptr());
    if (!iter) throw nb::python_error();
    PyObject *item;
    while ((item = PyIter_Next(iter))) {
        bool in_self = src.contains(item);
        bool ok      = true;
        if (!in_self)
            try { result.insert(item); }
            catch (...) { ok = false; }
        Py_DECREF(item);
        if (!ok) { Py_DECREF(iter); throw; }
    }
    Py_DECREF(iter);
    if (PyErr_Occurred()) throw nb::python_error();
    return result;
}

/* Compute __hash__ compatible with frozenset. */
static Py_hash_t compute_hash(const InsOrderedSetImpl &impl) {
    PyObject *fs = PyFrozenSet_New(nullptr);
    if (!fs) throw nb::python_error();
    for (size_t i = 0; i < impl.keys.size(); ++i) {
        if (!impl.alive[i]) continue;
        if (PySet_Add(fs, impl.keys[i]) < 0) {
            Py_DECREF(fs);
            throw nb::python_error();
        }
    }
    Py_hash_t h = PyObject_Hash(fs);
    Py_DECREF(fs);
    if (h == -1) throw nb::python_error();
    return h;
}

/* Build __repr__ string. */
static std::string build_repr(const InsOrderedSetImpl &impl,
                               const char *cls_name) {
    if (impl.used == 0)
        return std::string(cls_name) + "()";

    std::ostringstream oss;
    oss << cls_name << "({";
    bool first = true;
    for (size_t i = 0; i < impl.keys.size(); ++i) {
        if (!impl.alive[i]) continue;
        if (!first) oss << ", ";
        first = false;
        PyObject *r = PyObject_Repr(impl.keys[i]);
        if (!r) throw nb::python_error();
        const char *s = PyUnicode_AsUTF8(r);
        if (!s) { Py_DECREF(r); throw nb::python_error(); }
        oss << s;
        Py_DECREF(r);
    }
    oss << "})";
    return oss.str();
}

/* =========================================================================
   NB_MODULE
   =========================================================================*/

NB_MODULE(_insorderset_core, m) {
    m.doc() = "Insertion-ordered set C extension (nanobind).";

    /* ── Iterator ─────────────────────────────────────────────────────── */
    nb::class_<InsOrderedSetIterator>(m, "InsOrderedSetIterator")
        .def("__iter__", &InsOrderedSetIterator::__iter__,
             nb::rv_policy::reference)
        .def("__next__", &InsOrderedSetIterator::__next__);

    /* ── OrderedSet ───────────────────────────────────────────────────── */
    nb::class_<OrderedSet>(m, "OrderedSet",
        "Mutable insertion-ordered set.\n\n"
        "A drop-in replacement for :class:`set` that preserves the order in\n"
        "which elements were first inserted.  All elements must be hashable.\n\n"
        ":class:`OrderedSet` is **not** hashable (like :class:`set`).\n"
        "Use :class:`FrozenOrderedSet` for an immutable, hashable variant.")
        /* construction */
        .def("__init__",
             [](OrderedSet *self) { new (self) OrderedSet(); })
        .def("__init__",
             [](OrderedSet *self, nb::handle iterable) {
                 new (self) OrderedSet();
                 self->impl.extend_from(iterable.ptr());
             })

        /* core protocol */
        .def("__len__",  [](const OrderedSet &s) { return s.impl.used; })
        .def("__contains__",
             [](const OrderedSet &s, nb::handle key) {
                 return s.impl.contains(key.ptr());
             })
        .def("__iter__",
             [](const OrderedSet &s) {
                 return InsOrderedSetIterator(s.impl);
             })
        .def("__bool__", [](const OrderedSet &s) { return s.impl.used != 0; })
        .def("__repr__",
             [](const OrderedSet &s) { return build_repr(s.impl, "OrderedSet"); })
        .def("__hash__",
             [](const OrderedSet &) -> Py_hash_t {
                 PyErr_SetString(PyExc_TypeError,
                                 "unhashable type: 'OrderedSet'");
                 throw nb::python_error();
             })

        /* equality / ordering */
        .def("__eq__",
             [](const OrderedSet &s, nb::handle other) {
                 return impl_eq(s, other);
             })
        .def("__le__",
             [](const OrderedSet &s, nb::handle other) {
                 if (!PyAnySet_Check(other.ptr()) &&
                     !nb::isinstance<OrderedSet>(other) &&
                     !nb::isinstance<FrozenOrderedSet>(other)) {
                     return nb::not_implemented();
                 }
                 return impl_issubset(s, other);
             })
        .def("__lt__",
             [](const OrderedSet &s, nb::handle other) {
                 if (!PyAnySet_Check(other.ptr()) &&
                     !nb::isinstance<OrderedSet>(other) &&
                     !nb::isinstance<FrozenOrderedSet>(other)) {
                     return nb::not_implemented();
                 }
                 Py_ssize_t olen = PyObject_Length(other.ptr());
                 if (olen == -1) throw nb::python_error();
                 return s.impl.used < olen && impl_issubset(s, other);
             })
        .def("__ge__",
             [](const OrderedSet &s, nb::handle other) {
                 if (!PyAnySet_Check(other.ptr()) &&
                     !nb::isinstance<OrderedSet>(other) &&
                     !nb::isinstance<FrozenOrderedSet>(other)) {
                     return nb::not_implemented();
                 }
                 return impl_issuperset(s, other);
             })
        .def("__gt__",
             [](const OrderedSet &s, nb::handle other) {
                 if (!PyAnySet_Check(other.ptr()) &&
                     !nb::isinstance<OrderedSet>(other) &&
                     !nb::isinstance<FrozenOrderedSet>(other)) {
                     return nb::not_implemented();
                 }
                 Py_ssize_t olen = PyObject_Length(other.ptr());
                 if (olen == -1) throw nb::python_error();
                 return s.impl.used > olen && impl_issuperset(s, other);
             })

        /* mutation */
        .def("add",
             [](OrderedSet &s, nb::handle key) { s.impl.insert(key.ptr()); },
             "Add *key* to the set.  No-op if *key* is already present.")
        .def("discard",
             [](OrderedSet &s, nb::handle key) { s.impl.discard_key(key.ptr()); },
             "Remove *key* from the set if it is present.  No-op otherwise.")
        .def("remove",
             [](OrderedSet &s, nb::handle key) { s.impl.remove_key(key.ptr()); },
             "Remove *key* from the set.\n\n:raises KeyError: if *key* is not present.")
        .def("pop",
             [](OrderedSet &s) {
                 return nb::steal<nb::object>(s.impl.pop_last());
             },
             "Remove and return the last-inserted element.\n\n"
             ":raises KeyError: if the set is empty.")
        .def("clear",
             [](OrderedSet &s) { s.impl.clear_all(); },
             "Remove all elements from the set.")

        /* copy */
        .def("copy",
             [](const OrderedSet &s) { return OrderedSet{s.impl}; },
             "Return a shallow copy of the set.")

        /* set predicates */
        .def("isdisjoint",
             [](const OrderedSet &s, nb::handle other) {
                 return impl_isdisjoint(s, other);
             },
             "Return ``True`` if the set has no elements in common with *other*.")
        .def("issubset",
             [](const OrderedSet &s, nb::handle other) {
                 return impl_issubset(s, other);
             },
             "Test whether every element of the set is in *other*.")
        .def("issuperset",
             [](const OrderedSet &s, nb::handle other) {
                 return impl_issuperset(s, other);
             },
             "Test whether every element of *other* is in the set.")

        /* set operations returning OrderedSet */
        .def("difference",
             [](const OrderedSet &s, nb::args others) {
                 return OrderedSet{compute_difference(s.impl, others)};
             },
             "Return a new set with elements not in any of *others*.")
        .def("intersection",
             [](const OrderedSet &s, nb::args others) {
                 return OrderedSet{compute_intersection(s.impl, others)};
             },
             "Return a new set with elements common to the set and all *others*.")
        .def("union",
             [](const OrderedSet &s, nb::args others) {
                 return OrderedSet{compute_union(s.impl, others)};
             },
             "Return a new set with elements from the set and all *others*.")
        .def("symmetric_difference",
             [](const OrderedSet &s, nb::handle other) {
                 return OrderedSet{compute_symmdiff(s.impl, other)};
             },
             "Return a new set with elements in either set but not both.")

        /* in-place mutation */
        .def("difference_update",
             [](OrderedSet &s, nb::args others) {
                 s.impl = compute_difference(s.impl, others);
             },
             "Remove all elements found in *others* from the set in place.")
        .def("intersection_update",
             [](OrderedSet &s, nb::args others) {
                 s.impl = compute_intersection(s.impl, others);
             },
             "Retain only elements also found in all *others*, in place.")
        .def("symmetric_difference_update",
             [](OrderedSet &s, nb::handle other) {
                 s.impl = compute_symmdiff(s.impl, other);
             },
             "Update set to the symmetric difference with *other*, in place.")
        .def("update",
             [](OrderedSet &s, nb::args others) {
                 s.impl = compute_union(s.impl, others);
             },
             "Add all elements from *others* to the set in place.")

        /* binary operators */
        .def("__and__",
             [](const OrderedSet &s, nb::handle other) {
                 nb::args a = nb::cast<nb::args>(nb::make_tuple(other));
                 return OrderedSet{compute_intersection(s.impl, a)};
             })
        .def("__iand__",
             [](OrderedSet &s, nb::handle other) -> OrderedSet & {
                 nb::args a = nb::cast<nb::args>(nb::make_tuple(other));
                 s.impl = compute_intersection(s.impl, a);
                 return s;
             }, nb::rv_policy::reference)
        .def("__or__",
             [](const OrderedSet &s, nb::handle other) {
                 nb::args a = nb::cast<nb::args>(nb::make_tuple(other));
                 return OrderedSet{compute_union(s.impl, a)};
             })
        .def("__ior__",
             [](OrderedSet &s, nb::handle other) -> OrderedSet & {
                 nb::args a = nb::cast<nb::args>(nb::make_tuple(other));
                 s.impl = compute_union(s.impl, a);
                 return s;
             }, nb::rv_policy::reference)
        .def("__sub__",
             [](const OrderedSet &s, nb::handle other) {
                 nb::args a = nb::cast<nb::args>(nb::make_tuple(other));
                 return OrderedSet{compute_difference(s.impl, a)};
             })
        .def("__isub__",
             [](OrderedSet &s, nb::handle other) -> OrderedSet & {
                 nb::args a = nb::cast<nb::args>(nb::make_tuple(other));
                 s.impl = compute_difference(s.impl, a);
                 return s;
             }, nb::rv_policy::reference)
        .def("__xor__",
             [](const OrderedSet &s, nb::handle other) {
                 return OrderedSet{compute_symmdiff(s.impl, other)};
             })
        .def("__ixor__",
             [](OrderedSet &s, nb::handle other) -> OrderedSet & {
                 s.impl = compute_symmdiff(s.impl, other);
                 return s;
             }, nb::rv_policy::reference)

        /* pickling */
        .def("__reduce__",
             [](const OrderedSet &s) {
                 nb::list items;
                 for (size_t i = 0; i < s.impl.keys.size(); ++i)
                     if (s.impl.alive[i])
                         items.append(nb::borrow(s.impl.keys[i]));
                 return nb::make_tuple(nb::type<OrderedSet>(),
                                       nb::make_tuple(items));
             });

    /* ── FrozenOrderedSet ────────────────────────────────────────────── */
    nb::class_<FrozenOrderedSet>(m, "FrozenOrderedSet",
        "Immutable insertion-ordered set.\n\n"
        "A drop-in replacement for :class:`frozenset` that preserves the order\n"
        "in which elements were first inserted.  All elements must be hashable.\n\n"
        "The hash value is computed lazily on first access and then cached.  It\n"
        "is identical to the hash of the equivalent :class:`frozenset`, so a\n"
        ":class:`FrozenOrderedSet` can be used as a dictionary key or placed\n"
        "inside another set interchangeably with :class:`frozenset`.")
        /* construction */
        .def("__init__",
             [](FrozenOrderedSet *self) {
                 new (self) FrozenOrderedSet();
             })
        .def("__init__",
             [](FrozenOrderedSet *self, nb::handle iterable) {
                 new (self) FrozenOrderedSet();
                 self->impl.extend_from(iterable.ptr());
             })

        /* core protocol */
        .def("__len__",  [](const FrozenOrderedSet &s) { return s.impl.used; })
        .def("__contains__",
             [](const FrozenOrderedSet &s, nb::handle key) {
                 return s.impl.contains(key.ptr());
             })
        .def("__iter__",
             [](const FrozenOrderedSet &s) {
                 return InsOrderedSetIterator(s.impl);
             })
        .def("__bool__",
             [](const FrozenOrderedSet &s) { return s.impl.used != 0; })
        .def("__repr__",
             [](const FrozenOrderedSet &s) {
                 return build_repr(s.impl, "FrozenOrderedSet");
             })
        .def("__hash__",
             [](FrozenOrderedSet &s) {
                 if (!s.cached_hash.has_value())
                     s.cached_hash = compute_hash(s.impl);
                 return s.cached_hash.value();
             })

        /* equality / ordering */
        .def("__eq__",
             [](const FrozenOrderedSet &s, nb::handle other) {
                 return impl_eq(s, other);
             })
        .def("__le__",
             [](const FrozenOrderedSet &s, nb::handle other) {
                 return impl_issubset(s, other);
             })
        .def("__lt__",
             [](const FrozenOrderedSet &s, nb::handle other) {
                 Py_ssize_t olen = PyObject_Length(other.ptr());
                 if (olen == -1) throw nb::python_error();
                 return s.impl.used < olen && impl_issubset(s, other);
             })
        .def("__ge__",
             [](const FrozenOrderedSet &s, nb::handle other) {
                 return impl_issuperset(s, other);
             })
        .def("__gt__",
             [](const FrozenOrderedSet &s, nb::handle other) {
                 Py_ssize_t olen = PyObject_Length(other.ptr());
                 if (olen == -1) throw nb::python_error();
                 return s.impl.used > olen && impl_issuperset(s, other);
             })

        /* copy */
        .def("copy",
             [](const FrozenOrderedSet &s) {
                 return FrozenOrderedSet{s.impl, s.cached_hash};
             },
             "Return a shallow copy of the set.")

        /* set predicates */
        .def("isdisjoint",
             [](const FrozenOrderedSet &s, nb::handle other) {
                 return impl_isdisjoint(s, other);
             },
             "Return ``True`` if the set has no elements in common with *other*.")
        .def("issubset",
             [](const FrozenOrderedSet &s, nb::handle other) {
                 return impl_issubset(s, other);
             },
             "Test whether every element of the set is in *other*.")
        .def("issuperset",
             [](const FrozenOrderedSet &s, nb::handle other) {
                 return impl_issuperset(s, other);
             },
             "Test whether every element of *other* is in the set.")

        /* set operations returning FrozenOrderedSet */
        .def("difference",
             [](const FrozenOrderedSet &s, nb::args others) {
                 return FrozenOrderedSet{compute_difference(s.impl, others), {}};
             },
             "Return a new set with elements not in any of *others*.")
        .def("intersection",
             [](const FrozenOrderedSet &s, nb::args others) {
                 return FrozenOrderedSet{compute_intersection(s.impl, others), {}};
             },
             "Return a new set with elements common to the set and all *others*.")
        .def("union",
             [](const FrozenOrderedSet &s, nb::args others) {
                 return FrozenOrderedSet{compute_union(s.impl, others), {}};
             },
             "Return a new set with elements from the set and all *others*.")
        .def("symmetric_difference",
             [](const FrozenOrderedSet &s, nb::handle other) {
                 return FrozenOrderedSet{compute_symmdiff(s.impl, other), {}};
             },
             "Return a new set with elements in either set but not both.")

        /* binary operators */
        .def("__and__",
             [](const FrozenOrderedSet &s, nb::handle other) {
                 nb::args a = nb::cast<nb::args>(nb::make_tuple(other));
                 return FrozenOrderedSet{compute_intersection(s.impl, a), {}};
             })
        .def("__or__",
             [](const FrozenOrderedSet &s, nb::handle other) {
                 nb::args a = nb::cast<nb::args>(nb::make_tuple(other));
                 return FrozenOrderedSet{compute_union(s.impl, a), {}};
             })
        .def("__sub__",
             [](const FrozenOrderedSet &s, nb::handle other) {
                 nb::args a = nb::cast<nb::args>(nb::make_tuple(other));
                 return FrozenOrderedSet{compute_difference(s.impl, a), {}};
             })
        .def("__xor__",
             [](const FrozenOrderedSet &s, nb::handle other) {
                 return FrozenOrderedSet{compute_symmdiff(s.impl, other), {}};
             })

        /* pickling */
        .def("__reduce__",
             [](const FrozenOrderedSet &s) {
                 nb::list items;
                 for (size_t i = 0; i < s.impl.keys.size(); ++i)
                     if (s.impl.alive[i])
                         items.append(nb::borrow(s.impl.keys[i]));
                 return nb::make_tuple(nb::type<FrozenOrderedSet>(),
                                       nb::make_tuple(items));
             });
}
