from quantax.graph.data import GraphData
from quantax.core.glob import TraceData
import rustworkx

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
    graph = rustworkx.PyDiGraph(
        multigraph=False,
        node_count_hint=num_nodes + 1,
        edge_count_hint=num_edges + 1,
    )
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

