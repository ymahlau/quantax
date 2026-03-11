from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, get_args

import jax
import jax.numpy as jnp

from quantax.tracing.glob import (
    FunctionTransformNode,
    GlobalReplayData,
    get_current_node,
    get_global_replay_data,
    global_trace_context,
    node_context,
    register_node_input_output,
    register_node_pointer,
    replay_context,
)
from quantax.core.typing import AnyArrayLike, ShapedArrayLike
from quantax.core.utils import get_all_closure_vars
from quantax.tracing.graph import create_graph_from_trace
from quantax.tracing.optimization import solve_scale_assignment
from quantax.tracing.replay import get_replay_function
from quantax.tracing.tracer import UnitfulTracer
from quantax.unitful.unitful import Unitful


@dataclass(kw_only=True)
class JitTransformNode(FunctionTransformNode):
    is_outermost: bool
    jit_kwargs: dict[str, Any]

    def replay_node(
        self,
        *args,
        **kwargs,
    ):
        replay_data = get_global_replay_data()
        assert replay_data is not None
        cur_graph_data = replay_data.graph_data_dict[self.id][0]
        replay_fn = get_replay_function(
            graph_data=cur_graph_data,
            trace_args=self.trace_args,
            trace_kwargs=self.trace_kwargs,
            trace_result=self.output,
        )

        result = get_jit_original()(replay_fn, **self.jit_kwargs)(*args, **kwargs)

        return result





def convert_to_jax(data):
    converted_data = jax.tree.map(
        lambda x: jnp.asarray(x) if not isinstance(x, jax.Array | UnitfulTracer) else x,
        data,
    )
    return converted_data


def _trace_node(node: FunctionTransformNode, fn: Callable, fn_args, fn_kwargs):
    op_kwargs = parse_arg_kwargs(fn_args, fn_kwargs)
    node.op_kwargs = op_kwargs
    node.trace_args = fn_args
    node.trace_kwargs = fn_kwargs
    with node_context(node):
        trace_result = trace_fn(
            fn=fn,
            fn_transform_node=node,
            fn_args=fn_args,
            fn_kwargs=fn_kwargs,
            input_tracer_list=list(op_kwargs.values()),
        )
    node.output = trace_result
    return trace_result


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
            op_kwargs={},
            is_outermost=(not has_tracer),
            jit_kwargs=self.jit_kwargs,
        )

        if not has_tracer:
            # this is the outermost jit, we start tracing now
            with global_trace_context() as global_data:
                fn_args = convert_to_tracer(args)
                fn_kwargs = convert_to_tracer(kwargs)
                trace_result = _trace_node(node, self.fun, fn_args, fn_kwargs)

            # global_data still valid here (object persists after context exit)
            global_data.fn_transform_nodes[node.id] = node
        else:
            register_node_pointer(node)
            fn_args = args
            fn_kwargs = kwargs
            trace_result = _trace_node(node, self.fun, fn_args, fn_kwargs)
            result_leaves = jax.tree.leaves(trace_result, is_leaf=lambda x: isinstance(x, UnitfulTracer))
            register_tracer_for_current_context(result_leaves)
            register_node_input_output(node)
            return trace_result

        # if this was outermost jit, we are done tracing; optimize unitful scale assignment
        scale_assignment = solve_scale_assignment(
            global_data=global_data,
        )

        # compute topological ordering for every node
        data_dict = {
            n.id: [create_graph_from_trace(td) for td in n.fn_tracers] 
            for n in global_data.fn_transform_nodes.values()
        }

        # replay the function
        global_replay_data = GlobalReplayData(
            graph_data_dict=data_dict,
            scale_assignment=scale_assignment,
        )
        with replay_context(global_replay_data):
            result = node.replay_node(*args, **kwargs)

        return result


def get_jit_original() -> Callable:
    if hasattr(jax, "_orig_jit"):
        return jax._orig_jit
    return jax.jit


def jit(fun: Callable, **kwargs) -> UnitfulJitWrapped:
    custom_wrapper = UnitfulJitWrapped(fun=fun, jit_kwargs=kwargs)
    return custom_wrapper
