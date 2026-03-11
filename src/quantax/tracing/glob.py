from __future__ import annotations
from quantax.tracing.nodes import FunctionTransformNode, GlobalTraceData, TraceData, GlobalReplayData, ScaleAssignment, OperatorNode

from abc import abstractmethod
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from quantax.tracing.graph import create_graph_from_trace
import jax
from rustworkx import PyDiGraph

from quantax.core.typing import AnyArrayLike
from quantax.tracing.tracer import UnitfulTracer
from quantax.unitful.unitful import Unitful


_current_node: ContextVar[FunctionTransformNode | None] = ContextVar("current_node", default=None)
_current_trace_data: ContextVar[TraceData | None] = ContextVar("current_trace_data", default=None)
_global_trace_data: ContextVar[GlobalTraceData | None] = ContextVar("global_trace_data", default=None)
_global_replay_data: ContextVar[GlobalReplayData | None] = ContextVar("global_replay_data", default=None)


def get_current_node() -> FunctionTransformNode | None:
    return _current_node.get()


def get_current_trace_data() -> TraceData | None:
    return _current_trace_data.get()


def get_global_trace_data() -> GlobalTraceData | None:
    return _global_trace_data.get()


def get_global_replay_data() -> GlobalReplayData | None:
    return _global_replay_data.get()


@contextmanager
def global_trace_context():
    assert _global_trace_data.get() is None, "Trace end not called properly"
    new_data = GlobalTraceData()
    token = _global_trace_data.set(new_data)
    try:
        yield new_data
    finally:
        _global_trace_data.reset(token)


@contextmanager
def fn_trace_context():
    new_data = TraceData()
    token = _current_trace_data.set(new_data)
    try:
        yield new_data
    finally:
        _current_trace_data.reset(token)


@contextmanager
def node_context(new_node: FunctionTransformNode):
    token = _current_node.set(new_node)
    try:
        yield
    finally:
        _current_node.reset(token)


@contextmanager
def replay_context(
    scale_assignment: ScaleAssignment,
    global_trace_data: GlobalTraceData,
):
    assert _global_replay_data.get() is None, "Replay end not called properly before starting new replay"
    
    # compute topological ordering for every node
    data_dict = {
        n.id: [create_graph_from_trace(td) for td in n.fn_tracers] 
        for n in global_trace_data.fn_transform_nodes.values()
    }

    # set replay data struct
    global_replay_data = GlobalReplayData(
        graph_data_dict=data_dict,
        scale_assignment=scale_assignment,
    )
    token = _global_replay_data.set(global_replay_data)
    
    try:
        yield
    finally:
        _global_replay_data.reset(token)


def register_node_full(node: OperatorNode):
    register_node_pointer(node)
    register_node_input(node)
    register_node_output(node)


def register_node_input_output(node: OperatorNode):
    register_node_input(node)
    register_node_output(node)


def register_node_pointer(node: OperatorNode) -> None:
    global_td = get_global_trace_data()
    current_td = get_current_trace_data()
    assert global_td is not None
    assert node.id == -1, f"node has already been registered: {node}"
    node.id = len(global_td)
    if not isinstance(node, FunctionTransformNode):
        global_td.pure_operator_nodes[node.id] = node
    else:
        global_td.fn_transform_nodes[node.id] = node
    if current_td is not None:
        current_td.operator_nodes[node.id] = node


def register_node_input(node: OperatorNode):
    global_td = get_global_trace_data()
    current_td = get_current_trace_data()
    
    for t in node.op_kwargs.values():
        if current_td is not None:
            current_td.op_in_edges.append((t.id, node.id))
        # global graph does not consider function transform a node, only trace basic operations
        if not isinstance(node, FunctionTransformNode):
            assert global_td is not None
            global_td.op_in_edges.append((t.id, node.id))


def register_node_output(node: OperatorNode):
    global_td = get_global_trace_data()
    current_td = get_current_trace_data()
    trace_leaves = jax.tree.leaves(node.output)
    
    for t in trace_leaves:
        if current_td is not None:
            current_td.op_out_edges.append((node.id, t.id))
        # global graph does not consider function transform a node, only trace basic operations
        if not isinstance(node, FunctionTransformNode):
            assert global_td is not None
            global_td.op_out_edges.append((node.id, t.id))


def register_tracer(t: UnitfulTracer):
    global_td = get_global_trace_data()
    current_td = get_current_trace_data()
    assert global_td is not None
    t_idx = len(global_td)
    t.id = t_idx
    global_td.tracer_nodes[t.id] = t
    if current_td is not None:
        current_td.tracer_nodes[t.id] = t


def register_tracer_for_current_context(tracer_list: list[UnitfulTracer]):
    current_td = get_current_trace_data()
    assert current_td is not None
    for t in tracer_list:
        if t.id not in current_td.tracer_nodes:
            current_td.tracer_nodes[t.id] = t
