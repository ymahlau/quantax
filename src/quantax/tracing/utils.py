from typing import Any, get_args

import jax
import numpy as np

from quantax.core.constants import MAX_STATIC_OPTIMIZED_SIZE
from quantax.core.jax import is_traced
from quantax.core.typing import AnyArrayLike, StaticArrayLike
from quantax.tracing.glob import get_global_trace_data
from quantax.tracing.tracer import UnitfulTracer
from quantax.tracing.types import AnyUnitType
from quantax.unitful.unitful import Unitful


def convert_to_tracer(data):
    """
    Converts given pytree to tracers
    """

    def _conversion_helper(x: AnyArrayLike | Unitful | UnitfulTracer) -> UnitfulTracer:
        # if we already have a tracer, just return the tracer
        if isinstance(x, UnitfulTracer):
            return x

        static_unitful = get_static_operand(x)
        return UnitfulTracer(
            unit=x.unit if isinstance(x, Unitful) else None,
            static_unitful=static_unitful,
            value=x,
        )

    # IMPORTANT: we cannot use parallel tree_map here, because Tracer creation is linked to global list with an id.
    # race conditions may occur
    leaves, treedef = jax.tree.flatten(
        tree=data,
        is_leaf=lambda x: isinstance(x, (Unitful, UnitfulTracer)),
    )
    converted_leaves = []
    for leaf in leaves:
        if isinstance(leaf, tuple(list(get_args(AnyArrayLike)) + [Unitful])):
            converted_leaves.append(_conversion_helper(leaf))
        else:
            converted_leaves.append(leaf)
    converted_data = jax.tree.unflatten(treedef, converted_leaves)

    return converted_data


def convert_input(val: AnyUnitType) -> AnyUnitType:
    global_data = get_global_trace_data()
    # if we are currently not tracing, return value as is
    if global_data is None:
        assert not isinstance(val, UnitfulTracer), f"operation got tracer input, outside of tracing context: {val}"
        return val

    # tracer and standard values are left as is.
    if not isinstance(val, (Unitful, jax.Array)):
        return val

    # if we are tracing and input is jax array / Unitful, convert to unitful
    val_orig = val
    if isinstance(val, jax.Array):
        assert not is_traced(val), f"jax tracer detected during quantax tracing. This should never happen: {val}"
        val = Unitful(val=val)

    return UnitfulTracer(
        unit=None if isinstance(val_orig, jax.Array) else val.unit,
        static_unitful=get_static_operand(val),
        value=val_orig,
    )


def get_arg_kwarg_tracer_list(args, kwargs) -> list[UnitfulTracer]:
    return list(get_arg_kwarg_tracer_dict(args, kwargs).values())


def get_arg_kwarg_tracer_dict(args, kwargs) -> dict[str, UnitfulTracer]:
    op_kwargs = {}
    arg_leaves, arg_treedef = jax.tree.flatten(args, is_leaf=lambda x: isinstance(x, UnitfulTracer))
    for idx, la in enumerate(arg_leaves):
        assert isinstance(la, UnitfulTracer)
        op_kwargs[f"a_{idx}"] = la
    kwarg_leaves, kwarg_treedef = jax.tree.flatten(kwargs, is_leaf=lambda x: isinstance(x, UnitfulTracer))
    for idx, lk in enumerate(kwarg_leaves):
        assert isinstance(lk, UnitfulTracer)
        op_kwargs[f"k_{idx}"] = lk
    return op_kwargs


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
