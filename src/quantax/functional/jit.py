from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp

from quantax.core import glob
from quantax.core.glob import (
    FunctionTransformNode,
    GlobalReplayData,
    end_global_trace_data,
    init_global_trace_data,
    register_node_input_output,
    register_node_pointer,
    register_tracer_for_current_context,
    update_data_node_end,
    update_data_node_start,
    update_data_replay_end,
    update_data_replay_start,
)
from quantax.core.typing import AnyArrayLike, ShapedArrayLike
from quantax.core.utils import get_all_closure_vars
from quantax.functional.utils import parse_arg_kwargs, trace_fn
from quantax.tracing.graph import GraphData, create_graph_from_trace
from quantax.tracing.optimization import solve_scale_assignment
from quantax.tracing.replay import get_replay_function
from quantax.unitful.tracer import UnitfulTracer
from quantax.unitful.unitful import Unitful
from quantax.unitful.utils import check_jax_unitful_tracer_type, get_static_operand


@dataclass(kw_only=True)
class JitTransformNode(FunctionTransformNode):
    is_outermost: bool
    jit_kwargs: dict[str, Any]

    def replay_node(
        self,
        *args,
        **kwargs,
    ):
        assert glob.GLOBAL_REPLAY_DATA is not None
        cur_graph_data = glob.GLOBAL_REPLAY_DATA.graph_data_dict[self.id][0]
        replay_fn = get_replay_function(
            graph_data=cur_graph_data,
            trace_args=self.trace_args,
            trace_kwargs=self.trace_kwargs,
            trace_result=self.output,
        )

        orig_jit = get_jit_original()
        result = orig_jit(replay_fn, **self.jit_kwargs)(*args, *kwargs)

        return result


def convert_to_tracer(data):
    """
    Converts given pytree to tracers
    """

    def _conversion_helper(x: AnyArrayLike | Unitful) -> UnitfulTracer:
        if isinstance(x, ShapedArrayLike):
            sd = jax.ShapeDtypeStruct(shape=x.shape, dtype=x.dtype)
        else:
            sd = None

        static_unitful = get_static_operand(x)
        return UnitfulTracer(
            unit=x.unit if isinstance(x, Unitful) else None,
            val_shape_dtype=sd,
            static_unitful=static_unitful,
            value=x,
        )

    # IMPORTANT: we cannot use parallel tree_map here, because Tracer creation is linked to global list with an id.
    # race conditions may occur
    leaves, treedef = jax.tree.flatten(tree=data, is_leaf=lambda x: isinstance(x, Unitful))
    converted_leaves = []
    for leaf in leaves:
        if isinstance(leaf, AnyArrayLike | Unitful):
            converted_leaves.append(_conversion_helper(leaf))
        else:
            converted_leaves.append(leaf)
    converted_data = jax.tree.unflatten(treedef, converted_leaves)

    return converted_data


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
        _, has_unitful, has_tracer = check_jax_unitful_tracer_type(data=(args, kwargs, get_all_closure_vars(self.fun)))

        # if only standard jax arrays are involved, run standard jit
        if not has_unitful:
            return get_jit_original()(self.fun)(*args, **kwargs)

        node = JitTransformNode(
            op_name="jit",
            parent=glob.CURRENT_NODE,
            op_kwargs={},
            is_outermost=(not has_tracer),
            jit_kwargs=self.jit_kwargs,
        )

        if not has_tracer:
            # this is the outermost jit, we start tracing now
            init_global_trace_data()
            fn_args = convert_to_tracer(args)
            fn_kwargs = convert_to_tracer(kwargs)
        else:
            register_node_pointer(node)
            fn_args = args
            fn_kwargs = kwargs

        # update op kwarg entry of node (not possible before due to dependencies)
        op_kwargs = parse_arg_kwargs(fn_args, fn_kwargs)
        node.op_kwargs = op_kwargs
        node.trace_args = fn_args
        node.trace_kwargs = fn_kwargs

        # start node operation
        prev_node = update_data_node_start(node)
        trace_result = trace_fn(
            fn=self.fun,
            fn_transform_node=node,
            fn_args=fn_args,
            fn_kwargs=fn_kwargs,
            input_tracer_list=list(op_kwargs.values()),
        )

        # signal node end
        cur_node = update_data_node_end(prev_node)
        result_leaves = jax.tree.leaves(trace_result, is_leaf=lambda x: isinstance(x, UnitfulTracer))
        assert cur_node == node
        node.output = trace_result

        # if this was not the outermost jit, go back to tracing rest
        if has_tracer:
            register_tracer_for_current_context(result_leaves)
            register_node_input_output(node)
            return trace_result

        # if this was outermost jit, we are done tracing
        global_data = end_global_trace_data()
        global_data.fn_transform_nodes[node.id] = node

        # optimize unitful scale assignment
        scale_assignment = solve_scale_assignment(
            trace_args=fn_args,
            trace_kwargs=fn_kwargs,
            trace_output=trace_result,
            trace_data=global_data,
        )

        # compute topological ordering for every node
        data_dict: dict[int, list[GraphData]] = {}
        for n in global_data.fn_transform_nodes.values():
            cur_list = []
            for td in n.fn_tracers:
                graph_data = create_graph_from_trace(td)
                cur_list.append(graph_data)
            data_dict[n.id] = cur_list

        # replay the function
        global_replay_data = GlobalReplayData(
            graph_data_dict=data_dict,
            scale_assignment=scale_assignment,
        )
        update_data_replay_start(global_replay_data)
        result = node.replay_node(
            *args,
            **kwargs,
        )
        update_data_replay_end()

        return result


def get_jit_original() -> Callable:
    if hasattr(jax, "_orig_jit"):
        return jax._orig_jit
    return jax.jit


def jit(fun: Callable, **kwargs) -> UnitfulJitWrapped:
    custom_wrapper = UnitfulJitWrapped(fun=fun, jit_kwargs=kwargs)
    return custom_wrapper
