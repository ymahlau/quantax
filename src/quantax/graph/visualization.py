from rustworkx.visualization import mpl_draw

from quantax.graph.data import GraphData
from quantax.unitful.tracer import OperatorNode, UnitfulTracer


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
