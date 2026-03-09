from __future__ import annotations

from typing import Any, Callable

import jax
import numpy as np

from quantax.core.constants import MAX_STATIC_OPTIMIZED_SIZE
from quantax.core.glob import (
    FunctionTransformNode,
    register_tracer_for_current_context,
    update_data_fn_end,
    update_data_fn_start,
)
from quantax.core.typing import AnyArrayLike, StaticArrayLike
from quantax.core.utils import is_traced
from quantax.functional.artificial import noop
from quantax.unitful.tracer import UnitfulTracer
from quantax.unitful.unitful import Unitful


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


def parse_arg_kwargs(args, kwargs) -> dict[str, UnitfulTracer]:
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


def trace_fn(
    fn: Callable, fn_transform_node: FunctionTransformNode, fn_args, fn_kwargs, input_tracer_list: list[UnitfulTracer]
):
    prev_data = update_data_fn_start()
    register_tracer_for_current_context(input_tracer_list)
    trace_result = fn(*fn_args, **fn_kwargs)
    # TODO: technically it is only necessary to copy the output if it is an unchanged input tracer. This could reduce variable count in MILP
    result_copy = jax.tree.map(
        lambda x: noop(x), trace_result, is_leaf=lambda x: isinstance(x, (Unitful, UnitfulTracer))
    )

    cur_data = update_data_fn_end(prev_data, result=result_copy)
    fn_transform_node.fn_tracers.append(cur_data)

    return result_copy
