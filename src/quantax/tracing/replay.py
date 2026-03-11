from __future__ import annotations

from typing import Any, Callable

import jax

from quantax.tracing.glob import FunctionTransformNode, OperatorNode, get_global_replay_data
from quantax.core.typing import AnyArrayLike
from quantax.functional.collection import FUNCTION_DICT
from quantax.tracing.graph import GraphData
from quantax.tracing.tracer import UnitfulTracer
from quantax.unitful.unitful import Unitful


def get_replay_function(
    graph_data: GraphData,
    trace_args: Any,
    trace_kwargs: dict[str, Any],
    trace_result: Any,
) -> Callable:
    args_trace_leaves, args_trace_treedef = jax.tree.flatten(
        tree=trace_args, is_leaf=lambda x: isinstance(x, UnitfulTracer)
    )
    kwargs_trace_leaves, kwargs_trace_treedef = jax.tree.flatten(
        tree=trace_kwargs, is_leaf=lambda x: isinstance(x, UnitfulTracer)
    )
    result_trace_leaves, result_trace_treedef = jax.tree.flatten(
        tree=trace_result, is_leaf=lambda x: isinstance(x, UnitfulTracer)
    )

    # define the actual replay function
    def replay_func(*args, **kwargs):
        local_value_dict: dict[int, Unitful | AnyArrayLike] = {}
        args_leaves, args_treedef = jax.tree.flatten(args, lambda x: isinstance(x, Unitful))
        kwargs_leaves, kwargs_treedef = jax.tree.flatten(kwargs, lambda x: isinstance(x, Unitful))

        # sanity checks
        replay_data = get_global_replay_data()
        assert replay_data is not None
        assert args_treedef == args_trace_treedef
        assert kwargs_treedef == kwargs_trace_treedef
        assert len(args_leaves) == len(args_trace_leaves)
        assert len(kwargs_leaves) == len(kwargs_trace_leaves)
        global_value_dict = replay_data.value_dict
        scale_assignment = replay_data.scale_assignment

        # add inputs to value dictionary
        for leaf, t in zip(args_leaves, args_trace_leaves):
            if isinstance(leaf, Unitful):
                best_scale = scale_assignment.tracer_scales[t.id]
                leaf = leaf.set_fixed_scale(best_scale)
            else:
                # sanity check
                assert t.id not in scale_assignment.tracer_scales
            local_value_dict[t.id] = leaf
            if t.id not in global_value_dict:
                global_value_dict[t.id] = leaf
        # repeat above for kwargs
        for leaf, t in zip(kwargs_leaves, kwargs_trace_leaves):
            if isinstance(leaf, Unitful):
                best_scale = scale_assignment.tracer_scales[t.id]
                leaf = leaf.set_fixed_scale(best_scale)
            else:
                # sanity check
                assert t.id not in scale_assignment.tracer_scales
            local_value_dict[t.id] = leaf
            if t.id not in global_value_dict:
                global_value_dict[t.id] = leaf

        # follow topological ordering in computational graph and replay computation of each node
        for node_idx in graph_data.ordering:
            cur_node = graph_data.graph[node_idx]

            if isinstance(cur_node, UnitfulTracer):
                # Either the tracer is an input and already in local value dictionary
                # or a closure variable from outer context. In that case add to local dictionary
                if cur_node.id not in local_value_dict:
                    if cur_node.id in global_value_dict:
                        # this is a closure value from outer function which was traced
                        local_value_dict[cur_node.id] = global_value_dict[cur_node.id]
                    else:
                        # this is a closure/constant value which did not appear before
                        assert cur_node.value is not None
                        cur_value = cur_node.value
                        if isinstance(cur_value, Unitful):
                            # set the scale of the unitful value
                            best_scale = scale_assignment.tracer_scales[cur_node.id]
                            cur_value = cur_value.set_fixed_scale(best_scale)
                        else:
                            # sanity check
                            assert cur_node.id not in scale_assignment.tracer_scales
                        local_value_dict[cur_node.id] = cur_value
                        global_value_dict[cur_node.id] = cur_value

            elif isinstance(cur_node, OperatorNode):
                # perform operation
                if isinstance(cur_node, FunctionTransformNode):
                    # function transform node needs args and kwargs because it may make a difference (e.g. in jit)
                    # if input is located in args or kwargs
                    cur_args_leaves, cur_args_treedef = jax.tree.flatten(
                        tree=cur_node.trace_args,
                        is_leaf=lambda x: isinstance(x, UnitfulTracer),
                    )
                    cur_kwargs_leaves, cur_kwargs_treedef = jax.tree.flatten(
                        tree=cur_node.trace_kwargs,
                        is_leaf=lambda x: isinstance(x, UnitfulTracer),
                    )

                    # collect args
                    args_values = []
                    for idx, args_leaf in enumerate(cur_args_leaves):
                        if args_leaf.id in local_value_dict:
                            cur_val = local_value_dict[args_leaf.id]
                        elif args_leaf.id in global_value_dict:
                            cur_val = global_value_dict[args_leaf.id]
                        else:
                            raise Exception(f"value not found: {args_leaf}")
                        # add scale adjustment factor
                        if isinstance(cur_val, Unitful):
                            cur_key = (f"a_{idx}", args_leaf.id, cur_node.id)
                            scale_offset = scale_assignment.node_input_transforms[cur_key]
                            best_scale = scale_assignment.tracer_scales[args_leaf.id]
                            assert cur_val.scale == best_scale
                            cur_val = cur_val.add_scale_offset(scale_offset)
                        else:
                            # sanity check
                            assert args_leaf.id not in scale_assignment.tracer_scales
                        args_values.append(cur_val)

                    # collect kwargs
                    kwargs_values = []
                    for kwargs_leaf in cur_kwargs_leaves:
                        if kwargs_leaf.id in local_value_dict:
                            cur_val = local_value_dict[kwargs_leaf.id]
                        elif kwargs_leaf.id in global_value_dict:
                            cur_val = global_value_dict[kwargs_leaf.id]
                        else:
                            raise Exception(f"value not found: {kwargs_leaf}")
                        # add scale adjustment factor
                        if isinstance(cur_val, Unitful):
                            cur_key = (f"a_{idx}", kwargs_leaf.id, cur_node.id)
                            scale_offset = scale_assignment.node_input_transforms[cur_key]
                            best_scale = scale_assignment.tracer_scales[kwargs_leaf.id]
                            assert cur_val.scale == best_scale
                            cur_val = cur_val.add_scale_offset(scale_offset)
                        else:
                            # sanity check
                            assert kwargs_leaf.id not in scale_assignment.tracer_scales
                        kwargs_values.append(cur_val)

                    # construct final args and kwargs
                    cur_args = jax.tree.unflatten(cur_args_treedef, args_values)
                    cur_kwargs = jax.tree.unflatten(cur_kwargs_treedef, kwargs_values)

                    # call replay function from the current node
                    cur_result = cur_node.replay_node(*cur_args, **cur_kwargs)

                else:
                    # standard operation: just execute the function
                    input_values = {}
                    for k, v in cur_node.op_kwargs.items():
                        cur_val = local_value_dict[v.id]
                        # add scale adjustment factor
                        if isinstance(cur_val, Unitful):
                            cur_key = (k, v.id, cur_node.id)
                            scale_offset = scale_assignment.node_input_transforms[cur_key]
                            best_scale = scale_assignment.tracer_scales[v.id]
                            assert cur_val.scale == best_scale
                            cur_val = cur_val.add_scale_offset(scale_offset)
                        else:
                            # sanity check
                            assert v.id not in scale_assignment.tracer_scales
                        input_values[k] = cur_val

                    # exucute basic operation
                    cur_op_fn = FUNCTION_DICT[cur_node.op_name]
                    cur_result = cur_op_fn(**input_values)

                # parse results
                trace_result_leaves, trace_result_treedef = jax.tree.flatten(
                    tree=cur_node.output,
                    is_leaf=lambda x: isinstance(x, UnitfulTracer),
                )
                result_leaves, result_treedef = jax.tree.flatten(
                    tree=cur_result,
                    is_leaf=lambda x: isinstance(x, Unitful),
                )
                assert trace_result_treedef == result_treedef
                for tl, rl in zip(trace_result_leaves, result_leaves):
                    # add output offset to newly created unitfuls
                    if isinstance(rl, Unitful):
                        scale_offset = scale_assignment.tracer_pre_transforms[tl.id]
                        best_scale = scale_assignment.tracer_scales[tl.id]
                        assert rl.scale == best_scale - scale_offset
                        rl = rl.add_scale_offset(scale_offset)
                    else:
                        # sanity check
                        assert tl.id not in scale_assignment.tracer_scales

                    assert tl.id not in local_value_dict, "internal error, please report"
                    local_value_dict[tl.id] = rl
                    # if node was function transform, then it will already have registered the output tracer in the
                    # global dictionary
                    if not isinstance(cur_node, FunctionTransformNode):
                        assert tl.id not in global_value_dict, "internal error, please report"
                        global_value_dict[tl.id] = rl
            else:
                raise Exception(f"invalid node type: {cur_node}")

        # parse result of function eval
        result_list = []
        for t in result_trace_leaves:
            result_list.append(local_value_dict[t.id])
        result = jax.tree.unflatten(result_treedef, result_list)
        return result

    return replay_func
