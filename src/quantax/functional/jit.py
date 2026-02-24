from __future__ import annotations
from quantax.tracing.graph import create_graph_from_trace
from dataclasses import dataclass

from functools import partial
from typing import Any, Callable

import jax
import jax.numpy as jnp

from quantax.core import glob
from quantax.core.glob import TraceData, FunctionTransformNode, init_global_trace_data, update_data_node_start, update_data_node_end, end_global_trace_data, update_data_replay_start, update_data_replay_end, update_data_fn_start, update_data_fn_end, register_node_pointer, register_node_input_output, register_tracer_for_current_context
from quantax.core.typing import AnyArrayLike, ShapedArrayLike
from quantax.core.utils import get_all_closure_vars
from quantax.functional.utils import check_jax_unitful_tracer_type, get_static_operand, parse_arg_kwargs, trace_fn
from quantax.tracing.optimization import solve_scale_assignment
from quantax.tracing.replay import replay_execution
from quantax.unitful.tracer import UnitfulTracer
from quantax.unitful.unitful import Unitful


@dataclass(kw_only=True)
class JitTransformNode(FunctionTransformNode):
    is_outermost: bool

    def replay_node(self, *args, **kwargs):

        pass


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
    for l in leaves:
        if isinstance(l, AnyArrayLike | Unitful):
            converted_leaves.append(_conversion_helper(l))
        else:
            converted_leaves.append(l)
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
        # self.function_cache: dict[int, Callable] = {}

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

        # start node operation
        prev_node = update_data_node_start(node)
        result = trace_fn(
            fn=self.fun,
            fn_transform_node=node,
            fn_args=fn_args,
            fn_kwargs=fn_kwargs,
            input_tracer_list=list(op_kwargs.values()),
        )

        # signal node end
        cur_node = update_data_node_end(prev_node)
        result_leaves = jax.tree.leaves(result, is_leaf=lambda x: isinstance(x, UnitfulTracer))
        assert cur_node == node
        node.output = result

        # if this was not the outermost jit, go back to tracing rest
        if has_tracer:
            register_tracer_for_current_context(result_leaves)
            register_node_input_output(node)
            return result

        # if this was outermost jit, we are done tracing
        global_data = end_global_trace_data()
        global_data.fn_transform_nodes[node.id] = node

        # optimize unitful scale assignment
        scale_assignment = solve_scale_assignment(
            trace_args=fn_args,
            trace_kwargs=fn_kwargs,
            trace_output=result,
            trace_data=global_data,
        )

        # replay the function
        update_data_replay_start(scale_assignment)

        # compute topological ordering for every node
        for n in global_data.fn_transform_nodes.values():
            for td in n.fn_tracers:
                graph, graph_idx_to_node_id, node_id_to_graph_idx = create_graph_from_trace(td)
                a = 1

        update_data_replay_end()

        # return final result



        # # glob.CURRENT_NODE = tree_node

        # # if we are already tracing, continue tracing on lower level
        # if has_tracer:
        #     result = self.fun(*args, **kwargs)
        #     glob.CURRENT_NODE = tree_node.parent  # go back to previous meta node
        #     if glob.CURRENT_NODE is not None:
        #         glob.CURRENT_NODE.children.append(tree_node)
        #     node.output_tracer = result
        #     return result

        # # if we are not yet tracing, initialize the tracing mechanism and continue with final solve+replay afterwards
        # # assert glob.TRACE_DATA is None
        # # glob.TRACE_DATA = TraceData(special_tree=tree_node)
        # trace_args = convert_to_tracer(args)
        # trace_kwargs = convert_to_tracer(kwargs)

        # # call the function
        # trace_result = self.fun(*trace_args, **trace_kwargs)

        

        # # replay jax functions for jitting
        # a = 1

        # partial_exec_fn = partial(
        #     replay_execution,
        #     trace_result=trace_result,
        #     scale_assignment=scale_assignment,
        #     trace_data=glob.TRACE_DATA,
        # )
        # jitted_replay_fn = jax.jit(partial_exec_fn, **self.jit_kwargs)
        # result = jitted_replay_fn(args, kwargs)

        # del glob.TRACE_DATA
        # glob.TRACE_DATA = None

        # return result


def get_jit_original() -> Callable:
    if hasattr(jax, "_orig_jit"):
        return jax._orig_jit
    return jax.jit


def jit(fun: Callable, **kwargs) -> UnitfulJitWrapped:
    custom_wrapper = UnitfulJitWrapped(fun=fun, jit_kwargs=kwargs)
    return custom_wrapper

    # conv_args = convert_to_jax(args)
    # conv_kwargs = convert_to_jax(kwargs)
    # function_hash = hash((
    #     hash_abstract_unitful_pytree(args),
    #     hash_abstract_unitful_pytree(kwargs),
    # ))
    # if we already compiled before, just call cached function
    # if function_hash in self.function_cache:
    #     cached_fn = self.function_cache[function_hash]
    #     result = cached_fn(conv_args, conv_kwargs)
    #     return result
    # self.function_cache[function_hash] = jitted_replay_fn
