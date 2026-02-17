from __future__ import annotations
from functools import partial
from quantax.tracing.replay import replay_execution
from quantax.tracing.optimization import solve_scale_assignment
from typing import Any, Callable

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from quantax.core import glob
from quantax.core.glob import TraceData
from quantax.core.typing import AnyArrayLike, ShapedArrayLike, StaticArrayLike
from quantax.functional.utils import get_static_operand, hash_abstract_unitful_pytree
from quantax.unitful.tracer import UnitfulTracer
from quantax.core.unit import EMPTY_UNIT, Unit
from quantax.unitful.unitful import Unitful


def convert_to_tracer(data):
    """
    Converts given pytree to tracers
    """

    def _conversion_helper(x: AnyArrayLike | Unitful) -> UnitfulTracer:
        if  isinstance(x, ShapedArrayLike):
            sd = jax.ShapeDtypeStruct(shape=x.shape,dtype=x.dtype)
        else:
            sd = None
        
        static_unitful = get_static_operand(x)
        return UnitfulTracer(
            unit=x.unit if isinstance(x, Unitful) else EMPTY_UNIT,
            val_shape_dtype=sd,
            static_unitful=static_unitful,
            value=x,
        )
    
    # IMPORTANT: we cannot use parallel tree_map here, because Tracer creation is linked to global list with an id.
    # race conditions may occur
    leaves, treedef = jax.tree.flatten(
        tree=data, 
        is_leaf=lambda x: isinstance(x, Unitful)
    )
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
        lambda x: jnp.asarray(x) if not isinstance(x, jax.Array) else x,
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
        self.function_cache: dict[int, Callable] = {}

    def __call__(self, *args, **kwargs):
        conv_args = convert_to_jax(args)
        conv_kwargs = convert_to_jax(kwargs)
        function_hash = hash((
            hash_abstract_unitful_pytree(conv_args),
            hash_abstract_unitful_pytree(conv_kwargs),
        ))
        # if we already compiled before, just call cached function
        if function_hash in self.function_cache:
            cached_fn = self.function_cache[function_hash]
            result = cached_fn(conv_args, conv_kwargs)
            return result
        
        # start tracing
        glob.IS_UNITFUL_TRACING = True
        glob.TRACE_DATA = TraceData()

        trace_args = convert_to_tracer(conv_args)
        trace_kwargs = convert_to_tracer(conv_kwargs)

        trace_result = self.fun(*trace_args, **trace_kwargs)
        glob.IS_UNITFUL_TRACING = False
        
        # optimize unitful scale assignment
        scale_assignment: dict[int | tuple[str, int, int], int] = solve_scale_assignment(
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
        result = jitted_replay_fn(conv_args, conv_kwargs)
        self.function_cache[function_hash] = jitted_replay_fn
        
        del glob.TRACE_DATA
        
        return result


def jit(fun: Callable, **kwargs) -> UnitfulJitWrapped:
    custom_wrapper = UnitfulJitWrapped(fun=fun, jit_kwargs=kwargs)
    return custom_wrapper
