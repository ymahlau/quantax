from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from quantax.unitful.tracer import OperatorNode, UnitfulTracer

STATIC_OPTIM_STOP_FLAG: bool = False

IS_UNITFUL_TRACING: bool = False


@dataclass(kw_only=True)
class TraceData:
    nodes: list[OperatorNode | UnitfulTracer] = field(default_factory=list)
    node_in_edges: list[tuple[int, int, Any]] = field(default_factory=list)
    node_out_edges: list[tuple[int, int, Any]] = field(default_factory=list)


TRACE_DATA = TraceData()


def register_node(node: OperatorNode) -> None:
    global TRACE_DATA
    node_idx = len(TRACE_DATA.nodes)
    TRACE_DATA.nodes.append(node)
    # input edges
    for t in node.args.values():
        TRACE_DATA.node_in_edges.append((t.id, node_idx, None))
    # output edges
    for t in node.output_tracer:
        TRACE_DATA.node_out_edges.append((node_idx, t.id, None))


def register_tracer(t: UnitfulTracer) -> int:
    global TRACE_DATA
    t_idx = len(TRACE_DATA.nodes)
    TRACE_DATA.nodes.append(t)
    return t_idx
