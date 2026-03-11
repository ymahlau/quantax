from __future__ import annotations
from quantax.tracing.utils import get_static_operand, convert_to_tracer, get_arg_kwarg_tracer_list
from quantax.core.typing import AnyArrayLike, ShapedArrayLike
from quantax.unitful.utils import get_shape_dtype
from quantax.core.jax import is_traced
from quantax.tracing.types import AnyUnitType

from typing import Callable, TypeVarTuple, TypeVar, Unpack, ParamSpec, Generic, Any, get_args

import jax

from quantax.tracing.glob import (
    FunctionTransformNode,
    fn_trace_context,
    register_tracer_for_current_context, TraceData, get_global_trace_data, get_current_node,
)
from quantax.functional.artificial import noop
from quantax.tracing.tracer import UnitfulTracer
from quantax.unitful.unitful import Unitful

P = ParamSpec("P")
R = TypeVar("R")


class TracedFunction(Generic[P, R]):
    def __init__(self, fn: Callable[P, R]) -> None:
        self.fn = fn
        
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> tuple[R, TraceData]:
        
        with fn_trace_context() as trace_data:
            # convert function input to tracer, register for current context
            traced_args = convert_to_tracer(args)
            traced_kwargs = convert_to_tracer(kwargs)
            input_tracers = get_arg_kwarg_tracer_list(traced_args, traced_kwargs)
            register_tracer_for_current_context(input_tracers)
            
            # run the function
            result = self.fn(*traced_args, **traced_kwargs)
            
            # TODO: technically it is only necessary to copy the output if it is an unchanged input tracer. This could reduce variable count in MILP
            result_copy = jax.tree.map(
                f=lambda x: noop(x), 
                tree=result, 
                is_leaf=lambda x: isinstance(x, (Unitful, UnitfulTracer)),
            )
        
        # update trace data with collected information
        trace_data.trace_args = traced_args
        trace_data.trace_kwargs = traced_kwargs
        trace_data.output_tracer = result_copy
        
        return result_copy, trace_data


def trace(fn: Callable[P, R]) -> Callable[P, tuple[R, TraceData]]:
    traced = TracedFunction(fn=fn)
    return traced





# def parse_arg_kwargs(args, kwargs) -> dict[str, UnitfulTracer]:
#     op_kwargs = {}
#     arg_leaves, arg_treedef = jax.tree.flatten(args, is_leaf=lambda x: isinstance(x, UnitfulTracer))
#     for idx, la in enumerate(arg_leaves):
#         assert isinstance(la, UnitfulTracer)
#         op_kwargs[f"a_{idx}"] = la
#     kwarg_leaves, kwarg_treedef = jax.tree.flatten(kwargs, is_leaf=lambda x: isinstance(x, UnitfulTracer))
#     for idx, lk in enumerate(kwarg_leaves):
#         assert isinstance(lk, UnitfulTracer)
#         op_kwargs[f"k_{idx}"] = lk
#     return op_kwargs


# def trace_fn(
#     fn: Callable, fn_transform_node: FunctionTransformNode, fn_args, fn_kwargs, input_tracer_list: list[UnitfulTracer]
# ):
#     with fn_trace_context() as cur_data:
#         register_tracer_for_current_context(input_tracer_list)
#         trace_result = fn(*fn_args, **fn_kwargs)
#         # TODO: technically it is only necessary to copy the output if it is an unchanged input tracer. This could reduce variable count in MILP
#         result_copy = jax.tree.map(
#             lambda x: noop(x), trace_result, is_leaf=lambda x: isinstance(x, (Unitful, UnitfulTracer))
#         )
#         cur_data.output_tracer = result_copy

#     fn_transform_node.fn_tracers.append(cur_data)

#     return result_copy
