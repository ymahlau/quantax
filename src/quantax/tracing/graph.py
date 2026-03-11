from __future__ import annotations

import rustworkx
from rustworkx import PyDiGraph
from rustworkx.visualization import mpl_draw

from quantax.tracing.nodes import GraphData, OperatorNode, TraceData
from quantax.tracing.tracer import UnitfulTracer


def create_graph_from_trace(
    trace_data: TraceData,
) -> GraphData:
    num_nodes = len(trace_data)
    num_edges = len(trace_data.op_in_edges) + len(trace_data.op_out_edges)
    graph = PyDiGraph(
        multigraph=False,
        node_count_hint=num_nodes + 1,
        edge_count_hint=num_edges + 1,
    )
    node_indices = graph.add_nodes_from(trace_data.nodes)
    graph_idx_to_node_id: dict[int, int] = {i: n.id for (i, n) in zip(node_indices, trace_data.nodes)}
    node_id_to_graph_idx: dict[int, int] = {n.id: i for (i, n) in zip(node_indices, trace_data.nodes)}

    for parent_id, child_id in trace_data.op_in_edges + trace_data.op_out_edges:
        graph.add_edge(
            node_id_to_graph_idx[parent_id],
            node_id_to_graph_idx[child_id],
            None,
        )

    sorted_indices = list(rustworkx.topological_sort(graph))

    result = GraphData(
        graph=graph,
        graph_idx_to_node_id=graph_idx_to_node_id,
        node_id_to_graph_idx=node_id_to_graph_idx,
        trace_data=trace_data,
        ordering=sorted_indices,
    )

    return result


def get_label_from_node(node) -> str:
    if isinstance(node, UnitfulTracer):
        return node.id  # TODO: use variable name
    elif isinstance(node, OperatorNode):
        return node.op_name
    raise Exception("This should never happen")


def plot_graph(
    graph: PyDiGraph,
):
    mpl_draw(
        graph,
        labels=get_label_from_node,
        with_labels=True,
    )
