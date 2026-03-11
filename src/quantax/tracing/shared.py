from dataclasses import dataclass
from typing import Callable, Generic, ParamSpec, TypeVar

from quantax.functional.trace import trace
from quantax.tracing.glob import global_trace_context
from quantax.tracing.graph import create_graph_from_trace
from quantax.tracing.nodes import GlobalTraceData, GraphData, ScaleAssignment, TraceData
from quantax.tracing.optimization import solve_scale_assignment

P = ParamSpec("P")
R = TypeVar("R")


@dataclass(kw_only=True)
class TraceSolveResult(Generic[P, R]):
    trace_data: TraceData[P, R]
    graph_data: GraphData
    global_trace_data: GlobalTraceData
    scale_assignment: ScaleAssignment


class TraceSolveWrapper(Generic[P, R]):
    def __init__(self, fn: Callable[P, R]) -> None:
        self.fn = fn

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> tuple[R, TraceSolveResult[P, R]]:
        # trace
        with global_trace_context() as global_data:
            out, trace_data = trace(self.fn)(*args, **kwargs)

        # solve for optimal scales
        scale_assignment = solve_scale_assignment(global_data=global_data)
        graph_data = create_graph_from_trace(trace_data)

        result = TraceSolveResult(
            trace_data=trace_data,
            graph_data=graph_data,
            global_trace_data=global_data,
            scale_assignment=scale_assignment,
        )
        return out, result


def trace_and_solve(fn: Callable[P, R]) -> Callable[P, tuple[R, TraceSolveResult[P, R]]]:
    return TraceSolveWrapper(fn)
