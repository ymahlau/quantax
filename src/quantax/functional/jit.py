from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp

from quantax.core.utils import get_all_closure_vars
from quantax.functional.trace import trace
from quantax.tracing.glob import (
    FunctionTransformNode,
    get_current_node,
    global_trace_context,
    node_context,
    register_node_input_output,
    register_node_pointer,
    register_tracers_for_current_context,
    replay_context,
)
from quantax.tracing.optimization import solve_scale_assignment
from quantax.tracing.replay import get_replay_function
from quantax.tracing.tracer import UnitfulTracer
from quantax.tracing.utils import check_jax_unitful_tracer_type, get_arg_kwarg_tracer_dict


@dataclass(kw_only=True)
class JitTransformNode(FunctionTransformNode):
    is_outermost: bool
    jit_kwargs: dict[str, Any]

    def replay_node(
        self,
        *args,
        **kwargs,
    ):
        assert len(self.fn_trace_data) == 1
        cur_trace_data = self.fn_trace_data[0]
        replay_fn = get_replay_function(cur_trace_data)
        result = get_jit_original()(replay_fn, **self.jit_kwargs)(*args, **kwargs)

        return result


def convert_to_jax(data):
    converted_data = jax.tree.map(
        lambda x: jnp.asarray(x) if not isinstance(x, jax.Array | UnitfulTracer) else x,
        data,
    )
    return converted_data


class UnitfulJitWrapped:
    def __init__(
        self,
        fun: Callable,
        jit_kwargs: dict[str, Any],
    ):
        self.fun = fun
        self.jit_kwargs = jit_kwargs

    def __call__(self, *args, **kwargs):
        args = convert_to_jax(args)
        kwargs = convert_to_jax(kwargs)

        # do we have any untifuls in the computation involved
        data = (args, kwargs, get_all_closure_vars(self.fun))
        _, has_unitful, has_tracer = check_jax_unitful_tracer_type(data)

        # if only standard jax arrays are involved, run standard jit
        if not has_unitful:
            return get_jit_original()(self.fun, **self.jit_kwargs)(*args, **kwargs)

        node = JitTransformNode(
            op_name="jit",
            parent=get_current_node(),
            op_kwargs={},  # filled below in helper function
            is_outermost=(not has_tracer),
            jit_kwargs=self.jit_kwargs,
        )

        # helper function for tracing the current function
        def _trace_node_helper(*args, **kwargs):
            # trace
            with node_context(node):
                trace_result, trace_data = trace(self.fun, node)(*args, **kwargs)
            # fill node values
            node.trace_args = trace_data.trace_args
            node.trace_kwargs = trace_data.trace_kwargs
            node.output = trace_result
            op_kwargs = get_arg_kwarg_tracer_dict(trace_data.trace_args, trace_data.trace_args)
            node.op_kwargs = op_kwargs
            return trace_result, trace_data

        if not has_tracer:
            # this is the outermost jit, we start tracing now
            with global_trace_context() as global_data:
                trace_result, trace_data = _trace_node_helper(*args, **kwargs)

            # hardcode outermost operation with id -1
            assert node.id == -1
            global_data.fn_transform_nodes[node.id] = node
        else:
            register_node_pointer(node)
            trace_result, trace_data = _trace_node_helper(*args, **kwargs)
            result_leaves = jax.tree.leaves(trace_result, is_leaf=lambda x: isinstance(x, UnitfulTracer))
            register_tracers_for_current_context(result_leaves)
            register_node_input_output(node)
            return trace_result

        # if this was outermost jit, we are done tracing; optimize unitful scale assignment
        scale_assignment = solve_scale_assignment(global_data=global_data)

        with replay_context(scale_assignment, global_data):
            result = node.replay_node(*args, **kwargs)

        return result


def get_jit_original() -> Callable:
    if hasattr(jax, "_orig_jit"):
        return jax._orig_jit
    return jax.jit


def jit(fun: Callable, **kwargs) -> UnitfulJitWrapped:
    custom_wrapper = UnitfulJitWrapped(fun=fun, jit_kwargs=kwargs)
    return custom_wrapper
