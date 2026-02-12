from quantax.graph.visualization import plot_graph_data
from quantax.functional.utils import get_static_operand
from quantax.graph.creation import create_graph_from_trace
from quantax.core.glob import TraceData
from quantax.unitful.unit import EMPTY_UNIT
from quantax.unitful.tracer import UnitfulTracer
from quantax.core.typing import AnyArrayLike, ShapedArrayLike
from functools import partial
from typing import Any, Callable

import jax
import matplotlib.pyplot as plt

from quantax.core import glob
from quantax.core.utils import hash_abstract_pytree
from quantax.unitful.unitful import Unitful


# class UnitfulJitWrapped:
#     def __init__(
#         self,
#         fun: Callable,
#         jit_kwargs: dict[str, Any],
#     ):
#         self.fun = fun
#         self.jit_kwargs = jit_kwargs
#         self.jitted_fun = jax._orig_jit(  # type: ignore
#             fun, **jit_kwargs
#         )
#         self.cache = {}

#     def _call_from_cache(self, input_hash, *args, **kwargs):
#         to_materialise_mask = self.cache[input_hash]
#         # do actual jit compilation on function
#         optimized_result = self.jitted_fun(*args, **kwargs)
#         # materialise the results according to mask from unitful to arrays
#         materialised_result = jax.tree.map(
#             lambda x, y: x.array_materialise() if y else x,
#             optimized_result,
#             to_materialise_mask,
#             is_leaf=lambda x: isinstance(x, Unitful),
#         )
#         return materialised_result

#     def _compute_materialisation_mask(
#         self,
#         *args,
#         **kwargs,
#     ):
#         # first get shape dtypes without any array to unitful optimizations
#         # The functools.partial prevents jax from caching the function call
#         flags.STATIC_OPTIM_STOP_FLAG = True
#         desired_shape_dtype = jax.eval_shape(
#             jax._orig_jit(  # type: ignore
#                 partial(self.fun), **self.jit_kwargs
#             ),
#             *args,
#             **kwargs,
#         )
#         is_array_mask = jax.tree.map(
#             lambda x: isinstance(x, jax.ShapeDtypeStruct),
#             desired_shape_dtype,
#             is_leaf=lambda x: isinstance(x, Unitful),
#         )

#         # then get the result of the actual jit step
#         flags.STATIC_OPTIM_STOP_FLAG = False
#         optimized_shape_dtype = jax.eval_shape(
#             jax._orig_jit(  # type: ignore
#                 partial(self.fun), **self.jit_kwargs
#             ),
#             *args,
#             **kwargs,
#         )
#         is_unitful_mask = jax.tree.map(
#             lambda x: isinstance(x, Unitful),
#             optimized_shape_dtype,
#             is_leaf=lambda x: isinstance(x, Unitful),
#         )
#         # we need to convert all unitfuls back to arrays which are only arrays because of scale optimization
#         to_materialise_mask = jax.tree.map(
#             lambda x, y: x and y,
#             is_unitful_mask,
#             is_array_mask,
#         )
#         return to_materialise_mask

#     def __call__(self, *args, **kwargs):
#         # input hashing to avoid recompilation
#         input_hash = hash_abstract_pytree(tree=(args, kwargs))
#         if input_hash not in self.cache:
#             # compute the materialisation mask
#             to_materialise_mask = self._compute_materialisation_mask(*args, **kwargs)
#             self.cache[input_hash] = to_materialise_mask

#         return self._call_from_cache(input_hash, *args, **kwargs)

def convert_to_tracer(data):
    """
    Converts given pytree to tracers
    """
    def _conversion_helper(x: AnyArrayLike | Unitful) -> UnitfulTracer:
        sd = None if not isinstance(x, ShapedArrayLike) else  jax.ShapeDtypeStruct(
            shape=x.shape,
            dtype=x.dtype,
        )
        static_arr = get_static_operand(x)
        return UnitfulTracer(
            unit=x.unit if isinstance(x, Unitful) else EMPTY_UNIT,
            val_shape_dtype=sd,
            static_arr=static_arr,
        )
        
    converted_data = jax.tree.map(
        lambda x: _conversion_helper(x) if isinstance(x, AnyArrayLike | Unitful) else x,
        data,
        is_leaf=lambda x: isinstance(x, Unitful),
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
        glob.IS_UNITFUL_TRACING = True
        glob.TRACE_DATA = TraceData()
        
        trace_args = convert_to_tracer(args)
        trace_kwargs = convert_to_tracer(kwargs)
        
        trace_result = self.fun(*trace_args, **trace_kwargs)
        graph_data = create_graph_from_trace(
            args=args,
            kwargs=kwargs,
            trace_args=trace_args,
            trace_kwargs=trace_kwargs,
            trace_output=trace_result,
            trace_data=glob.TRACE_DATA,
        )
        del glob.TRACE_DATA
        glob.IS_UNITFUL_TRACING = False
        plot_graph_data(graph_data)
        plt.savefig("tmp.png")
        return
        a = 1
        pass


def jit(fun: Callable, **kwargs) -> UnitfulJitWrapped:
    custom_wrapper = UnitfulJitWrapped(fun=jax.tree_util.Partial(fun), jit_kwargs=kwargs)
    return custom_wrapper
