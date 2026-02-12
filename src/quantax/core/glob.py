from __future__ import annotations
from functools import cached_property

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from quantax.unitful.tracer import OperatorNode, UnitfulTracer

STATIC_OPTIM_STOP_FLAG: bool = False

IS_UNITFUL_TRACING: bool = False


@dataclass(kw_only=True)
class TraceData:
    operator_nodes: list[OperatorNode] = field(default_factory=list)
    tracer_nodes: list[UnitfulTracer] = field(default_factory=list)
    node_in_edges: list[tuple[int, int, Any]] = field(default_factory=list)
    node_out_edges: list[tuple[int, int, Any]] = field(default_factory=list)

    @property
    def nodes(self) -> list[OperatorNode | UnitfulTracer]:
        return self.operator_nodes + self.tracer_nodes
    
    def __len__(self) -> int:
        return len(self.operator_nodes) + len(self.tracer_nodes)

TRACE_DATA = TraceData()

def register_node(node: OperatorNode) -> None:
    global TRACE_DATA
    node.id = len(TRACE_DATA)
    TRACE_DATA.operator_nodes.append(node)
    # input edges
    for t in node.args.values():
        TRACE_DATA.node_in_edges.append((t.id, node.id, None))
    # output edges
    for t in node.output_tracer:
        TRACE_DATA.node_out_edges.append((node.id, t.id, None))


def register_tracer(t: UnitfulTracer) -> int:
    global TRACE_DATA
    t_idx = len(TRACE_DATA)
    TRACE_DATA.tracer_nodes.append(t)
    return t_idx
