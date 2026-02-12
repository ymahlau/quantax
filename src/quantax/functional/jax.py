from quantax.tracing.optimization import solve_scale_assignment
from typing import Any, Callable

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from quantax.core import glob
from quantax.core.glob import TraceData
from quantax.core.typing import AnyArrayLike, ShapedArrayLike, StaticArrayLike
from quantax.functional.utils import get_static_operand
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
        )

    converted_data = jax.tree.map(
        lambda x: _conversion_helper(x) if isinstance(x, AnyArrayLike | Unitful) else x,
        data,
        is_leaf=lambda x: isinstance(x, Unitful),
    )
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

    def __call__(self, *args, **kwargs):
        conv_args = convert_to_jax(args)
        conv_kwargs = convert_to_jax(kwargs)
        

        trace_args = convert_to_tracer(conv_args)
        trace_kwargs = convert_to_tracer(conv_kwargs)

        glob.IS_UNITFUL_TRACING = True
        glob.TRACE_DATA = TraceData()
        trace_result = self.fun(*trace_args, **trace_kwargs)
        glob.IS_UNITFUL_TRACING = False
        
        scale_assignment: dict[int | tuple[str, int, int], int] = solve_scale_assignment(
            args=args,
            kwargs=kwargs,
            trace_args=trace_args,
            trace_kwargs=trace_kwargs,
            trace_output=trace_result,
            trace_data=glob.TRACE_DATA,
        )
        
        del glob.TRACE_DATA
        
        
        
        
        # plot_graph_data(graph_data)
        # plt.savefig("tmp.png")
        return
        a = 1
        pass


def jit(fun: Callable, **kwargs) -> UnitfulJitWrapped:
    custom_wrapper = UnitfulJitWrapped(fun=jax.tree_util.Partial(fun), jit_kwargs=kwargs)
    return custom_wrapper
