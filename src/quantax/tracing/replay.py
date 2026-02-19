from __future__ import annotations

import jax

from quantax.core.glob import TraceData
from quantax.core.typing import AnyArrayLike
from quantax.functional.collection import FUNCTION_DICT
from quantax.unitful.tracer import UnitfulTracer
from quantax.unitful.unitful import Unitful


def replay_execution(
    jax_args,
    jax_kwargs,
    trace_result,
    trace_data: TraceData,
    scale_assignment: dict[tuple[int, bool] | tuple[str, int, int], int],
):
    value_dict: dict[int, Unitful | AnyArrayLike] = {
        t.id: t.value for t in trace_data.tracer_nodes if t.value is not None
    }

    for n in trace_data.operator_nodes:
        # get operator input
        input_kwargs = {}
        for k, v in n.args.items():
            if isinstance(v, UnitfulTracer):
                # use fixed scale computed beforehand as assignment
                base_scale = scale_assignment[v.id]
                scale_offset = scale_assignment[(k, v.id, n.id)]
                cur_unitful = value_dict[v.id]
                # the value here should always be a unitful, because non-unitfuls are not traced
                assert isinstance(cur_unitful, Unitful), "internal error, please report"
                assert cur_unitful.scale == base_scale
                converted_unitful = cur_unitful.add_scale_offset(scale_offset)
                input_kwargs[k] = converted_unitful
            else:
                input_kwargs[k] = v

        # call function
        cur_fn = FUNCTION_DICT[n.op_name]
        cur_result = cur_fn(**input_kwargs)

        # map outputs to corresponding tracers
        val_leaves, treedef = jax.tree.flatten(tree=cur_result, is_leaf=lambda x: isinstance(x, Unitful))
        trace_leaves, treedef2 = jax.tree.flatten(
            tree=n.output_tracer,
            is_leaf=lambda x: isinstance(x, Unitful | UnitfulTracer),
        )
        assert treedef == treedef2, "internal error, please report"
        for l, t in zip(val_leaves, trace_leaves):
            value_dict[t.id] = l

    # reconstruct the function results from all computed values
    result = jax.tree.map(
        f=lambda t: value_dict[t.id],
        tree=trace_result,
        is_leaf=lambda x: isinstance(x, UnitfulTracer),
    )
    return result
