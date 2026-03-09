from typing import Any

import jax
import numpy as np

from quantax.core.constants import MAX_STATIC_OPTIMIZED_SIZE
from quantax.core.jax import is_traced
from quantax.core.typing import AnyArrayLike, StaticArrayLike
from quantax.unitful.tracer import UnitfulTracer
from quantax.unitful.unitful import Unitful


def hash_abstract_unitful_pytree(
    data: Any,
) -> int:
    leaves, tree_def = jax.tree.flatten(
        tree=data,
        is_leaf=lambda x: isinstance(x, Unitful),
    )
    to_hash = []
    for leaf in leaves:
        if isinstance(leaf, Unitful):
            to_hash.append((leaf.scale, hash_abstract_arraylike(leaf.val)))
        elif isinstance(leaf, AnyArrayLike):
            to_hash.append(hash_abstract_arraylike(leaf))
        else:
            raise Exception(f"Invalid hash input: {leaf}")
    return hash(tuple(to_hash))


def hash_abstract_arraylike(v: AnyArrayLike):
    if hasattr(v, "shape") and hasattr(v, "dtype"):
        return hash((v.shape, str(v.dtype)))
    # non-shaped arrays are python scalars, which get same hash every time
    return 0


def check_jax_unitful_tracer_type(
    data: Any,
) -> tuple[bool, bool, bool]:
    # TODO: make more efficient by parallelising, avoiding python for loop
    leaves = jax.tree.leaves(
        tree=data,
        is_leaf=lambda x: isinstance(x, (Unitful, UnitfulTracer)),
    )
    has_jax, has_tracer, has_unitful = False, False, False
    for leaf in leaves:
        if isinstance(leaf, jax.Array):
            has_jax = True
        if isinstance(leaf, Unitful):
            has_unitful = True
            if isinstance(leaf.val, jax.Array):
                has_jax = True
        if isinstance(leaf, UnitfulTracer):
            has_unitful = True
            has_tracer = True
    return has_jax, has_unitful, has_tracer


def get_static_operand(
    x: UnitfulTracer | AnyArrayLike | Unitful,
) -> Unitful | None:
    if isinstance(x, Unitful):
        if is_traced(x.val):
            return None
        if isinstance(x.val, jax.Array):
            if x.val.size >= MAX_STATIC_OPTIMIZED_SIZE:
                return None
            return Unitful(val=np.asarray(x.val, copy=True), scale=x.scale, unit=x.unit)
        return x

    # Physical arraylike without a unit
    if isinstance(x, AnyArrayLike):
        if is_traced(x):
            return None
        if isinstance(x, jax.Array):
            if x.size >= MAX_STATIC_OPTIMIZED_SIZE:
                return None
            return Unitful(val=np.asarray(x, copy=True))
        assert isinstance(x, StaticArrayLike), "Internal error, please report"
        return Unitful(val=x)

    # tracer
    assert isinstance(x, UnitfulTracer), "Internal error, please report"
    return x.static_unitful
