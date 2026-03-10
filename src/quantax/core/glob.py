from __future__ import annotations

from abc import abstractmethod
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import jax
from rustworkx import PyDiGraph

from quantax.core.typing import AnyArrayLike

if TYPE_CHECKING:
    from quantax.unitful.tracer import UnitfulTracer
    from quantax.unitful.unitful import Unitful

_current_node: ContextVar[FunctionTransformNode | None] = ContextVar('current_node', default=None)
_current_trace_data: ContextVar[TraceData | None] = ContextVar('current_trace_data', default=None)
_global_trace_data: ContextVar[GlobalTraceData | None] = ContextVar('global_trace_data', default=None)
_global_replay_data: ContextVar[GlobalReplayData | None] = ContextVar('global_replay_data', default=None)


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
def replay_context(global_replay_data: GlobalReplayData):
    assert _global_replay_data.get() is None, "Replay end not called properly before starting new replay"
    token = _global_replay_data.set(global_replay_data)
    try:
        yield
    finally:
        _global_replay_data.reset(token)


@dataclass(kw_only=True)
class GlobalTraceData:
    pure_operator_nodes: dict[int, OperatorNode] = field(default_factory=dict)
    tracer_nodes: dict[int, UnitfulTracer] = field(default_factory=dict)
    node_in_edges: list[tuple[int, int]] = field(default_factory=list)
    node_out_edges: list[tuple[int, int]] = field(default_factory=list)
    fn_transform_nodes: dict[int, FunctionTransformNode] = field(default_factory=dict)

    @property
    def nodes(self) -> list[OperatorNode | UnitfulTracer]:
        return self.operator_nodes + list(self.tracer_nodes.values())

    @property
    def operator_nodes(self) -> list[OperatorNode]:
        return list(self.pure_operator_nodes.values()) + list(self.fn_transform_nodes.values())

    def __len__(self) -> int:
        return len(self.pure_operator_nodes) + len(self.tracer_nodes) + len(self.fn_transform_nodes)


@dataclass(kw_only=True)
class TraceData:
    output_tracer: Any = None  # struct of tracers
    operator_nodes: dict[int, OperatorNode] = field(default_factory=dict)
    tracer_nodes: dict[int, UnitfulTracer] = field(default_factory=dict)
    node_in_edges: list[tuple[int, int]] = field(default_factory=list)
    node_out_edges: list[tuple[int, int]] = field(default_factory=list)

    @property
    def nodes(self) -> list[OperatorNode | UnitfulTracer]:
        return list(self.operator_nodes.values()) + list(self.tracer_nodes.values())

    def __len__(self) -> int:
        return len(self.operator_nodes) + len(self.tracer_nodes)

    @property
    def children(self) -> list[FunctionTransformNode]:
        return [n for n in self.operator_nodes.values() if isinstance(n, FunctionTransformNode)]


@dataclass(kw_only=True)
class OperatorNode:
    op_name: str
    op_kwargs: dict[str, UnitfulTracer]
    output: Any = ()  # should be struct of tracers
    metadata: dict[str, Any] = field(default_factory=dict)
    id: int = field(default=-1, init=False)

    def __post_init__(self):
        # sanity checks
        from quantax.unitful.tracer import UnitfulTracer

        for t in jax.tree.leaves(self.output, is_leaf=lambda x: isinstance(x, UnitfulTracer)):
            assert isinstance(t, UnitfulTracer)

    def __eq__(self, other):
        if not isinstance(other, OperatorNode):
            return False
        return self.id == other.id


@dataclass(kw_only=True)
class FunctionTransformNode(OperatorNode):
    fn_tracers: list[TraceData] = field(default_factory=list)
    parent: FunctionTransformNode | None
    trace_args: Any = field(default=None)
    trace_kwargs: dict[str, Any] = field(default_factory=dict)

    @abstractmethod
    def replay_node(
        self,
        *args,
        **kwargs,
    ):
        pass


@dataclass(kw_only=True)
class ScaleAssignment:
    # scale exponents for every tracer
    tracer_scales: dict[int, int]

    # mapping from
    # - operation argument (e.g. 'x' is first argument to multiply)
    # - tracer id
    # - operation node id
    # to the corresponding scale transformation that needs to be performed. Zero means no transformation
    node_input_transforms: dict[tuple[str, int, int], int]

    # Sometimes, when a tracer is created as output of an operation, a scale transform already needs to be performed
    # These are listed here. Again zero means no transformation.
    tracer_pre_transforms: dict[int, int]


@dataclass(kw_only=True)
class GraphData:
    graph: PyDiGraph
    graph_idx_to_node_id: dict[int, int]
    node_id_to_graph_idx: dict[int, int]
    trace_data: TraceData
    ordering: list[int]


@dataclass(kw_only=True)
class GlobalReplayData:
    graph_data_dict: dict[int, list[GraphData]]
    scale_assignment: ScaleAssignment
    value_dict: dict[int, Unitful | AnyArrayLike] = field(default_factory=dict)


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
    assert current_td is not None
    assert node.id == -1, f"node has already been registered: {node}"
    node.id = len(global_td)
    if not isinstance(node, FunctionTransformNode):
        global_td.pure_operator_nodes[node.id] = node
    else:
        global_td.fn_transform_nodes[node.id] = node
    current_td.operator_nodes[node.id] = node


def register_node_input(node: OperatorNode):
    global_td = get_global_trace_data()
    current_td = get_current_trace_data()
    assert current_td is not None
    for t in node.op_kwargs.values():
        current_td.node_in_edges.append((t.id, node.id))
        # global graph does not consider function transform a node, only trace basic operations
        if not isinstance(node, FunctionTransformNode):
            assert global_td is not None
            global_td.node_in_edges.append((t.id, node.id))


def register_node_output(node: OperatorNode):
    global_td = get_global_trace_data()
    current_td = get_current_trace_data()
    assert current_td is not None
    assert global_td is not None
    trace_leaves = jax.tree.leaves(node.output)
    for t in trace_leaves:
        current_td.node_out_edges.append((node.id, t.id))
        # global graph does not consider function transform a node, only trace basic operations
        if not isinstance(node, FunctionTransformNode):
            global_td.node_out_edges.append((node.id, t.id))


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
