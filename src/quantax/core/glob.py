from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import jax

if TYPE_CHECKING:
    from quantax.unitful.tracer import OperatorNode, UnitfulTracer

STATIC_OPTIM_STOP_FLAG: bool = False

META_NODE: SpecialOpsTreeNode | None = None


@dataclass(kw_only=True)
class SpecialOpsTreeNode:
    op: OperatorNode
    parent: SpecialOpsTreeNode | None
    children: list[SpecialOpsTreeNode] = field(default_factory=list)


@dataclass(kw_only=True)
class TraceData:
    special_tree: SpecialOpsTreeNode
    operator_nodes: list[OperatorNode] = field(default_factory=list)
    tracer_nodes: list[UnitfulTracer] = field(default_factory=list)
    node_in_edges: list[tuple[int, int, Any]] = field(default_factory=list)
    node_out_edges: list[tuple[int, int, Any]] = field(default_factory=list)

    @property
    def nodes(self) -> list[OperatorNode | UnitfulTracer]:
        return self.operator_nodes + self.tracer_nodes

    def __len__(self) -> int:
        return len(self.operator_nodes) + len(self.tracer_nodes)


TRACE_DATA: TraceData | None = None


def register_node(node: OperatorNode) -> None:
    global TRACE_DATA
    assert TRACE_DATA is not None
    node.id = len(TRACE_DATA)
    TRACE_DATA.operator_nodes.append(node)
    # input edges
    for t in node.args.values():
        TRACE_DATA.node_in_edges.append((t.id, node.id, None))
    # output edges
    trace_leaves = jax.tree.leaves(node.output_tracer)
    for t in trace_leaves:
        TRACE_DATA.node_out_edges.append((node.id, t.id, None))


def register_tracer(t: UnitfulTracer) -> int:
    global TRACE_DATA
    assert TRACE_DATA is not None
    t_idx = len(TRACE_DATA)
    TRACE_DATA.tracer_nodes.append(t)
    return t_idx
