from typing import Any, get_args

import jax

from quantax.core.typing import AnyArrayLike
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
        elif isinstance(leaf, get_args(AnyArrayLike)):
            to_hash.append(hash_abstract_arraylike(leaf))
        else:
            raise Exception(f"Invalid hash input: {leaf}")
    return hash(tuple(to_hash))


def hash_abstract_arraylike(v: AnyArrayLike):
    if hasattr(v, "shape") and hasattr(v, "dtype"):
        return hash((v.shape, str(v.dtype)))
    # non-shaped arrays are python scalars, which get same hash every time
    return 0


def get_shape_dtype(
    val: AnyArrayLike | Unitful,
) -> jax.ShapeDtypeStruct | None:
    if isinstance(val, Unitful):
        val = val.val
    if not hasattr(val, "shape"):
        return None
    if not hasattr(val, "dtype"):
        return None
    return jax.ShapeDtypeStruct(val.shape, val.dtype)
