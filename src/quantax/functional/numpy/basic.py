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
from quantax.unitful.tracer import UnitfulTracer
from quantax.unitful.types import AnyUnitType
from quantax.unitful.unitful import Unitful


## Multiplication ###########################
def constraints_multiply(
    x: mathopt.Variable | None,
    y: mathopt.Variable | None,
    out: Any | None,
) -> list[mathopt.BoundedLinearExpression]:
    if out is None:
        assert x is None and y is None
        return []
    assert isinstance(out, mathopt.LinearSum)
    if x is None:
        assert y is not None
        return [y == out]
    if y is None:
        assert x is not None
        return [x == out]
    return [x + y == out]


def get_multiply_original():
    if hasattr(jax.numpy, "_orig_multiply"):
        return jax.numpy._orig_multiply
    return jax.numpy.multiply


@overload
def multiply(x: Unitful, y: Unitful) -> Unitful: ...


@overload
def multiply(x: AnyArrayLike, y: Unitful) -> Unitful: ...


@overload
def multiply(x: Unitful, y: AnyArrayLike) -> Unitful: ...


@overload
def multiply(x: int, y: int) -> int: ...


@overload
def multiply(x: int, y: float) -> float: ...


@overload
def multiply(x: float, y: int) -> float: ...


@overload
def multiply(x: float, y: float) -> float: ...


@overload
def multiply(x: complex, y: int | float) -> complex: ...


@overload
def multiply(x: int | float | complex, y: complex) -> complex: ...


@overload
def multiply(x: int | float | complex | np.number, y: np.ndarray) -> np.ndarray: ...


@overload
def multiply(x: np.ndarray, y: int | float | complex | np.number) -> np.ndarray: ...


@overload
def multiply(x: int | float | complex | np.number, y: jax.Array) -> jax.Array: ...


@overload
def multiply(x: jax.Array, y: int | float | complex | np.number) -> jax.Array: ...


@overload
def multiply(x: jax.Array, y: jax.Array) -> jax.Array: ...


@overload
def multiply(x: jax.Array, y: np.ndarray) -> jax.Array: ...


@overload
def multiply(x: np.ndarray, y: jax.Array) -> jax.Array: ...


@overload
def multiply(x: np.ndarray, y: np.ndarray) -> np.ndarray: ...


def multiply(x: AnyUnitType, y: AnyUnitType) -> AnyUnitType:
    # Unitful handling
    def _mul_unitful(x: Unitful, y: Unitful):
        new_unit = dim_after_multiplication(x.unit, y.unit)
        assert new_unit is not None
        new_val = x.val * y.val
        new_scale = x.scale + y.scale
        return Unitful(val=new_val, unit=new_unit, scale=new_scale)

    if isinstance(x, Unitful) and isinstance(y, Unitful):
        return _mul_unitful(x, y)
    elif isinstance(x, Unitful) and not isinstance(y, (Unitful | UnitfulTracer)):
        y_unitful = Unitful(val=y)
        return _mul_unitful(x, y_unitful)
    elif not isinstance(x, (Unitful | UnitfulTracer)) and isinstance(y, Unitful):
        x_unitful = Unitful(val=x)
        return _mul_unitful(x_unitful, y)

    # Tracer handling
    def _mul_tracer(x: UnitfulTracer, y: UnitfulTracer):
        new_unit = dim_after_multiplication(x.unit, y.unit)
        new_static_unitful = None
        if x.static_unitful is not None and y.static_unitful is not None:
            new_static_unitful = x.static_unitful * y.static_unitful
        result = UnitfulTracer(unit=new_unit, static_unitful=new_static_unitful)
        node = OperatorNode(
            op_name="multiply",
            op_kwargs={"x": x, "y": y},
            output=result,
        )
        register_node_full(node)
        return result

    if isinstance(x, UnitfulTracer) and isinstance(y, UnitfulTracer):
        return _mul_tracer(x, y)
    if isinstance(x, UnitfulTracer):
        assert isinstance(y, StaticArrayLike | Unitful)
        y_unitful = Unitful(val=y) if not isinstance(y, Unitful) else y
        tracer_unit = None if not isinstance(y, Unitful) else y.unit
        y_tracer = UnitfulTracer(unit=tracer_unit, static_unitful=y_unitful, value=y)
        return _mul_tracer(x, y_tracer)
    if isinstance(y, UnitfulTracer):
        assert isinstance(x, StaticArrayLike | Unitful)
        x_unitful = Unitful(val=x) if not isinstance(x, Unitful) else x
        tracer_unit = None if not isinstance(x, Unitful) else x.unit
        x_tracer = UnitfulTracer(unit=tracer_unit, static_unitful=x_unitful, value=x)
        return _mul_tracer(x_tracer, y)

    # any other array-like
    assert isinstance(x, AnyArrayLike), f"Invalid input type for multiply: {x}"
    assert isinstance(y, AnyArrayLike), f"Invalid input type for multiply: {y}"
    result = x * y
    assert isinstance(result, AnyArrayLike)
    return result