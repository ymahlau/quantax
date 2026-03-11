from __future__ import annotations

from typing import Callable, Generic, ParamSpec, TypeVar

import jax

from quantax.functional.artificial import noop
from quantax.tracing.glob import (
    FunctionTransformNode,
    TraceData,
    fn_trace_context,
    register_tracers_for_current_context,
)
from quantax.tracing.tracer import UnitfulTracer
from quantax.tracing.utils import convert_to_tracer, get_arg_kwarg_tracer_list
from quantax.unitful.unitful import Unitful

P = ParamSpec("P")
R = TypeVar("R")


class TracedFunction(Generic[P, R]):
    def __init__(
        self,
        fn: Callable[P, R],
        node: FunctionTransformNode | None = None,
    ) -> None:
        self.fn = fn
        self.node = node

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> tuple[R, TraceData[P, R]]:
        with fn_trace_context() as trace_data:
            # convert function input to tracer, register for current context
            traced_args = convert_to_tracer(args)
            traced_kwargs = convert_to_tracer(kwargs)
            input_tracers = get_arg_kwarg_tracer_list(traced_args, traced_kwargs)
            register_tracers_for_current_context(input_tracers)

            # run the function
            result = self.fn(*traced_args, **traced_kwargs)

            # TODO: technically it is only necessary to copy the output if it is an unchanged input tracer.
            # This could reduce variable count in MILP
            result_copy = jax.tree.map(
                f=lambda x: noop(x),
                tree=result,
                is_leaf=lambda x: isinstance(x, (Unitful, UnitfulTracer)),
            )

        # update trace data with collected information
        trace_data.trace_args = traced_args
        trace_data.trace_kwargs = traced_kwargs
        trace_data.output_tracer = result_copy

        # update node information
        if self.node is not None:
            self.node.fn_trace_data.append(trace_data)

        return result_copy, trace_data


def trace(
    fn: Callable[P, R],
    node: FunctionTransformNode | None = None,
) -> Callable[P, tuple[R, TraceData[P, R]]]:
    traced = TracedFunction(fn=fn, node=node)
    return traced


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
