from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Generic, ParamSpec, TypeVar

import jax
from rustworkx import PyDiGraph

from quantax.core.typing import AnyArrayLike
from quantax.tracing.tracer import UnitfulTracer
from quantax.unitful.unitful import Unitful


@dataclass(kw_only=True)
class GraphData:
    graph: PyDiGraph
    graph_idx_to_node_id: dict[int, int]
    node_id_to_graph_idx: dict[int, int]
    trace_data: TraceData
    ordering: list[int]


@dataclass(kw_only=True)
class GlobalReplayData:
    scale_assignment: ScaleAssignment
    value_dict: dict[int, Unitful | AnyArrayLike] = field(default_factory=dict)


@dataclass(kw_only=True)
class GlobalTraceData:
    pure_operator_nodes: dict[int, OperatorNode] = field(default_factory=dict)
    tracer_nodes: dict[int, UnitfulTracer] = field(default_factory=dict)
    op_in_edges: list[tuple[int, int]] = field(default_factory=list)
    op_out_edges: list[tuple[int, int]] = field(default_factory=list)
    fn_transform_nodes: dict[int, FunctionTransformNode] = field(default_factory=dict)

    @property
    def nodes(self) -> list[OperatorNode | UnitfulTracer]:
        return self.operator_nodes + list(self.tracer_nodes.values())

    @property
    def operator_nodes(self) -> list[OperatorNode]:
        return list(self.pure_operator_nodes.values()) + list(self.fn_transform_nodes.values())

    def __len__(self) -> int:
        return len(self.pure_operator_nodes) + len(self.tracer_nodes) + len(self.fn_transform_nodes)


P = ParamSpec("P")
R = TypeVar("R")


@dataclass(kw_only=True)
class TraceData(Generic[P, R]):
    output_tracer: Any = field(default=None)  # struct of tracers
    trace_args: Any = field(default=None)  # struct of tracers
    trace_kwargs: dict[str, Any] = field(default_factory=dict)  # struct of tracers
    operator_nodes: dict[int, OperatorNode] = field(default_factory=dict)
    tracer_nodes: dict[int, UnitfulTracer] = field(default_factory=dict)
    op_in_edges: list[tuple[int, int]] = field(default_factory=list)
    op_out_edges: list[tuple[int, int]] = field(default_factory=list)

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
        for t in jax.tree.leaves(self.output, is_leaf=lambda x: isinstance(x, UnitfulTracer)):
            assert isinstance(t, UnitfulTracer)

    def __eq__(self, other):
        if not isinstance(other, OperatorNode):
            return False
        return self.id == other.id


@dataclass(kw_only=True)
class FunctionTransformNode(OperatorNode, ABC):
    fn_trace_data: list[TraceData] = field(default_factory=list)
    op_kwargs: dict[str, UnitfulTracer]
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
