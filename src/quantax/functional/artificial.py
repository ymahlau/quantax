from __future__ import annotations

from typing import Any, overload

import jax
import numpy as np
from ortools.math_opt.python import mathopt

from quantax.core.glob import OperatorNode, register_node_full
from quantax.core.typing import AnyArrayLike, StaticArrayLike
from quantax.core.utils import (
    dim_after_multiplication,
)
from quantax.functional.utils import AnyUnitType
from quantax.unitful.tracer import UnitfulTracer
from quantax.unitful.unitful import Unitful


def constraints_noop(
    x: mathopt.Variable,
    out: Any,
) -> list[mathopt.BoundedLinearExpression]:
    assert isinstance(out, (mathopt.Variable, mathopt.LinearSum))
    return [x == out]


@overload
def noop(x: Unitful) -> Unitful: ...


@overload
def noop(x: int) -> int: ...


@overload
def noop(x: float) -> float: ...


@overload
def noop(x: complex) -> complex: ...


@overload
def noop(x: jax.Array) -> jax.Array: ...


@overload
def noop(x: np.ndarray) -> np.ndarray: ...


@overload
def noop(x: np.ndarray) -> np.ndarray: ...


def noop(x: AnyUnitType) -> AnyUnitType:
    # non-tracer
    if not isinstance(x, UnitfulTracer):
        return x
    
    # tracer
    x_cpy = UnitfulTracer(
        unit=x.unit,
        val_shape_dtype=x.val_shape_dtype,
        static_unitful=x.static_unitful,
        value=x.value,
    )
    node = OperatorNode(
        op_name="noop",
        op_kwargs={"x": x},
        output=x_cpy,
    )
    register_node_full(node)
    return x_cpy




