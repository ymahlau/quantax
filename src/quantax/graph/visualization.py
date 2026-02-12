from quantax.unitful.tracer import UnitfulTracer, OperatorNode
from quantax.graph.data import GraphData
from rustworkx.visualization import mpl_draw

def get_label_from_node(node) -> str:
    if isinstance(node, UnitfulTracer):
        return node.id  # TODO: use variable name
    elif isinstance(node, OperatorNode):
        return node.op_name
    raise Exception(f"This should never happen")


def plot_graph_data(
    data: GraphData,
):
    mpl_draw(
        data.graph,
        labels=get_label_from_node,
        with_labels=True,
    )


