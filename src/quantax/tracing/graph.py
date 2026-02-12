from rustworkx.visualization import mpl_draw
from quantax.unitful.tracer import UnitfulTracer, OperatorNode
from typing import Any
from rustworkx import PyDiGraph
from dataclasses import dataclass

from quantax.core.glob import TraceData

@dataclass(frozen=True, kw_only=True)
class GraphData:
    graph: PyDiGraph
    args: Any
    kwargs: Any
    trace_args: Any
    trace_kwargs: Any
    trace_output: Any


def create_graph_from_trace(
    args,
    kwargs,
    trace_args,
    trace_kwargs,
    trace_output,
    trace_data: TraceData,
) -> GraphData:
    num_nodes = len(trace_data.nodes)
    num_edges = len(trace_data.node_in_edges) + len(trace_data.node_out_edges)
    graph = PyDiGraph(
        multigraph=False,
        node_count_hint=num_nodes + 1,
        edge_count_hint=num_edges + 1,
    )
    # TODO: fix. This is wrong because order of nodes changed, need sorting
    graph.add_nodes_from(trace_data.nodes)
    graph.add_edges_from(trace_data.node_in_edges)
    graph.add_edges_from(trace_data.node_out_edges)

    return GraphData(
        graph=graph,
        args=args,
        kwargs=kwargs,
        trace_args=trace_args,
        trace_kwargs=trace_kwargs,
        trace_output=trace_output,
    )
    

def get_label_from_node(node) -> str:
    if isinstance(node, UnitfulTracer):
        return node.id  # TODO: use variable name
    elif isinstance(node, OperatorNode):
        return node.op_name
    raise Exception("This should never happen")


def plot_graph_data(
    data: GraphData,
):
    mpl_draw(
        data.graph,
        labels=get_label_from_node,
        with_labels=True,
    )
