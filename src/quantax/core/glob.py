from __future__ import annotations
from abc import abstractmethod

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

import jax

if TYPE_CHECKING:
    from quantax.unitful.tracer import UnitfulTracer

CURRENT_NODE: FunctionTransformNode | None = None
CURRENT_TRACE_DATA: TraceData | None = None
GLOBAL_DATA: GlobalTraceData | None = None
GLOBAL_SCALE_ASSIGNMENT: ScaleAssignment | None = None


@dataclass(kw_only=True)
class GlobalTraceData:
    operator_nodes: dict[int, OperatorNode] = field(default_factory=dict)
    tracer_nodes: dict[int, UnitfulTracer] = field(default_factory=dict)
    node_in_edges: list[tuple[int, int]] = field(default_factory=list)
    node_out_edges: list[tuple[int, int]] = field(default_factory=list)
    fn_transform_nodes: dict[int, FunctionTransformNode] = field(default_factory=dict)

    @property
    def nodes(self) -> list[OperatorNode | UnitfulTracer]:
        return list(self.operator_nodes.values()) + list(self.tracer_nodes.values())

    def __len__(self) -> int:
        return len(self.operator_nodes) + len(self.tracer_nodes) + len(self.fn_transform_nodes)


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
        if not isinstance(other, OperatorNode): return False
        return self.id == other.id


@dataclass(kw_only=True)
class FunctionTransformNode(OperatorNode):
    fn_tracers: list[TraceData] = field(default_factory=list)
    parent: FunctionTransformNode | None

    @abstractmethod
    def replay_node(self):
        pass


@dataclass(kw_only=True)
class ScaleAssignment:
    # scale exponents for every tracer
    tracer_scales: dict[int, int | float]
    
    # mapping from 
    # - operation argument (e.g. 'x' is first argument to multiply)
    # - tracer id
    # - operation node id
    # to the corresponding scale transformation that needs to be performed. Zero means no transformation
    node_input_transforms: dict[tuple[str, int, int], int]

    # Sometimes, when a tracer is created as output of an operation, a scale transform already needs to be performed
    # These are listed here. Again zero means no transformation.
    tracer_pre_transforms: dict[int, int]


def init_global_trace_data():
    global GLOBAL_DATA
    assert GLOBAL_DATA is None, "Trace end not called properly"
    GLOBAL_DATA = GlobalTraceData()


def end_global_trace_data() -> GlobalTraceData:
    global GLOBAL_DATA
    assert GLOBAL_DATA is not None, "Trace start not called properly"
    cur_data = GLOBAL_DATA
    GLOBAL_DATA = None
    return cur_data


def update_data_fn_start() -> TraceData | None:
    global CURRENT_TRACE_DATA
    prev_data = CURRENT_TRACE_DATA

    CURRENT_TRACE_DATA = TraceData()

    return prev_data


def update_data_fn_end(prev_data: TraceData | None, result: Any) -> TraceData:
    global CURRENT_TRACE_DATA

    finished_data = CURRENT_TRACE_DATA
    assert finished_data is not None, "fn start was not called before fn end"
    CURRENT_TRACE_DATA = prev_data
    finished_data.output_tracer = result

    return finished_data


def update_data_node_start(new_node: FunctionTransformNode):
    global CURRENT_NODE
    prev_node = CURRENT_NODE

    CURRENT_NODE = new_node

    return prev_node

def update_data_node_end(prev_node: FunctionTransformNode):
    global CURRENT_NODE
    cur_node = CURRENT_NODE
    CURRENT_NODE = prev_node
    return cur_node


def update_data_replay_start(scale_assignment: ScaleAssignment):
    global GLOBAL_SCALE_ASSIGNMENT
    assert GLOBAL_SCALE_ASSIGNMENT is None, "scale assignment end was not called properly before starting new replay"
    GLOBAL_SCALE_ASSIGNMENT = scale_assignment


def update_data_replay_end():
    global GLOBAL_SCALE_ASSIGNMENT
    assert GLOBAL_SCALE_ASSIGNMENT is not None, "scale assignment start was not called before end"
    GLOBAL_SCALE_ASSIGNMENT = None


def register_node_full(node: OperatorNode):
    register_node_pointer(node)
    register_node_input(node)
    register_node_output(node)

def register_node_input_output(node: OperatorNode):
    register_node_input(node)
    register_node_output(node)


def register_node_pointer(node: OperatorNode) -> None:
    global GLOBAL_DATA, CURRENT_TRACE_DATA
    assert GLOBAL_DATA is not None
    assert CURRENT_TRACE_DATA is not None
    assert node.id == -1, f"node has already been registered: {node}"
    node.id = len(GLOBAL_DATA)
    if not isinstance(node, FunctionTransformNode):
        GLOBAL_DATA.operator_nodes[node.id] = node
    else:
        GLOBAL_DATA.fn_transform_nodes[node.id] = node
    CURRENT_TRACE_DATA.operator_nodes[node.id] = node


def register_node_input(node: OperatorNode):
    global GLOBAL_DATA, CURRENT_TRACE_DATA
    assert CURRENT_TRACE_DATA is not None
    for t in node.op_kwargs.values():
        CURRENT_TRACE_DATA.node_in_edges.append((t.id, node.id))
        # global graph does not consider function transform a node, only trace basic operations
        if not isinstance(node, FunctionTransformNode):
            assert GLOBAL_DATA is not None
            GLOBAL_DATA.node_in_edges.append((t.id, node.id))


def register_node_output(node: OperatorNode):
    global  GLOBAL_DATA, CURRENT_TRACE_DATA
    assert CURRENT_TRACE_DATA is not None
    assert GLOBAL_DATA is not None
    trace_leaves = jax.tree.leaves(node.output)
    for t in trace_leaves:
        CURRENT_TRACE_DATA.node_out_edges.append((node.id, t.id))
        # global graph does not consider function transform a node, only trace basic operations
        if not isinstance(node, FunctionTransformNode):
            GLOBAL_DATA.node_out_edges.append((node.id, t.id))


def register_tracer(t: UnitfulTracer):
    global GLOBAL_DATA, CURRENT_TRACE_DATA
    assert GLOBAL_DATA is not None
    t_idx = len(GLOBAL_DATA)
    t.id = t_idx
    GLOBAL_DATA.tracer_nodes[t.id] = t
    if CURRENT_TRACE_DATA is not None:
        CURRENT_TRACE_DATA.tracer_nodes[t.id] = t


def register_tracer_for_current_context(tracer_list: list[UnitfulTracer]):
    assert CURRENT_TRACE_DATA is not None
    for t in tracer_list:
        if t.id not in CURRENT_TRACE_DATA.tracer_nodes:
            CURRENT_TRACE_DATA.tracer_nodes[t.id] = t
