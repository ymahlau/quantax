from __future__ import annotations

from typing import Callable

import jax

from quantax.core.glob import (
    FunctionTransformNode,
    register_tracer_for_current_context,
    update_data_fn_end,
    update_data_fn_start,
)
from quantax.functional.artificial import noop
from quantax.unitful.tracer import UnitfulTracer
from quantax.unitful.unitful import Unitful


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
