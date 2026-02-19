from typing import Callable, TypeVar

import jax
import numpy as np
from typing_extensions import TypeVarTuple, Unpack

from quantax.core import glob
from quantax.core.glob import register_node
from quantax.functional.utils import check_jax_unitful_tracer_type
from quantax.unitful.tracer import OperatorNode, UnitfulTracer
from quantax.unitful.unitful import Unitful

Ts = TypeVarTuple("Ts")
R = TypeVar("R")


def cond(
    pred: bool | jax.Array | np.bool_ | Unitful,
    true_fun: Callable[Unpack[Ts], R],
    false_fun: Callable[Unpack[Ts], R],
    *operands: Unpack[Ts],
) -> R:
    # test for jax/unitful components
    has_jax, has_unitful, has_tracer = check_jax_unitful_tracer_type(data=(pred, operands))

    # CASE 1: traced implementation
    if has_tracer:
        node = OperatorNode(
            op_name="cond",
            args={"pred": pred, "true_fun": true_fun, "false_fun": false_fun, "*operands": operands},
        )
        prev_meta_node = glob.META_NODE
        glob.META_NODE = node
        true_result = true_fun(*operands)
        false_result = false_fun(*operands)
        glob.META_NODE = prev_meta_node

        result = UnitfulTracer(unit=new_unit, parent=node, static_unitful=new_static_unitful)
        node.output_tracer = result
        register_node(node)
        return result

    # CASE 2: unitful eager implementation
    if has_unitful:
        pass

    # CASE 3: standard jax call
    return jax.lax.cond(
        pred,
        true_fun,
        false_fun,
        *operands,
    )
