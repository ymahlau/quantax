from __future__ import annotations

from functools import partial
from typing import Any, Callable

import jax
import jax.numpy as jnp

from quantax.core import glob
from quantax.core.glob import SpecialOpsTreeNode, TraceData
from quantax.core.typing import AnyArrayLike, ShapedArrayLike
from quantax.core.utils import get_all_closure_vars
from quantax.functional.utils import check_jax_unitful_tracer_type, get_static_operand
from quantax.tracing.optimization import solve_scale_assignment
from quantax.tracing.replay import replay_execution
from quantax.unitful.tracer import OperatorNode, UnitfulTracer
from quantax.unitful.unitful import Unitful


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
            return jit_original()(self.fun)(*args, **kwargs)

        # if we are not already tracing, initialize tracing
        node = OperatorNode(
            op_name="jit",
            args={"*args": args, "**kwargs": kwargs},
        )
        tree_node = SpecialOpsTreeNode(
            op=node,
            parent=glob.META_NODE,
        )
        glob.META_NODE = tree_node

        # if we are already tracing, continue tracing on lower level
        if has_tracer:
            result = self.fun(*args, **kwargs)
            glob.META_NODE = tree_node.parent  # go back to previous meta node
            if glob.META_NODE is not None:
                glob.META_NODE.children.append(tree_node)
            node.output_tracer = result
            return result

        # if we are not yet tracing, initialize the tracing mechanism and continue with final solve+replay afterwards
        assert glob.TRACE_DATA is None
        glob.TRACE_DATA = TraceData(special_tree=tree_node)
        trace_args = convert_to_tracer(args)
        trace_kwargs = convert_to_tracer(kwargs)

        # call the function
        trace_result = self.fun(*trace_args, **trace_kwargs)

        # optimize unitful scale assignment
        scale_assignment: dict[tuple[int, bool] | tuple[str, int, int], int] = solve_scale_assignment(
            args=args,
            kwargs=kwargs,
            trace_args=trace_args,
            trace_kwargs=trace_kwargs,
            trace_output=trace_result,
            trace_data=glob.TRACE_DATA,
        )

        # replay jax functions for jitting
        partial_exec_fn = partial(
            replay_execution,
            trace_result=trace_result,
            scale_assignment=scale_assignment,
            trace_data=glob.TRACE_DATA,
        )
        jitted_replay_fn = jax.jit(partial_exec_fn, **self.jit_kwargs)
        result = jitted_replay_fn(args, kwargs)

        del glob.TRACE_DATA
        glob.TRACE_DATA = None

        return result


def jit_original() -> Callable:
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
