from __future__ import annotations
from quantax.tracing.utils import convert_input, get_static_operand
from quantax.unitful.utils import get_shape_dtype

from typing import Any, get_args, overload

import jax
import numpy as np
from ortools.math_opt.python import mathopt

from quantax.tracing.glob import OperatorNode, register_node_full
from quantax.core.typing import AnyArrayLike, StaticArrayLike
from quantax.core.unit import Unit
from quantax.core.utils import (
    dim_after_multiplication,
    handle_different_scales,
)
from quantax.tracing.tracer import UnitfulTracer
from quantax.tracing.types import AnyUnitType
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
    x = convert_input(x)
    y = convert_input(y)
    
    # Unitful handling
    def _mul_unitful(x: Unitful, y: Unitful) -> Unitful:
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
            new_static_unitful = _mul_unitful(x.static_unitful, y.static_unitful)
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
        y_tracer = UnitfulTracer(
            unit=None if not isinstance(y, Unitful) else y.unit, 
            static_unitful=get_static_operand(y), 
            value=y, 
            val_shape_dtype=get_shape_dtype(y),
        )
        return _mul_tracer(x, y_tracer)
    if isinstance(y, UnitfulTracer):
        assert isinstance(x, StaticArrayLike | Unitful)
        x_tracer = UnitfulTracer(
            unit=None if not isinstance(x, Unitful) else x.unit, 
            static_unitful=get_static_operand(x), 
            value=x, 
            val_shape_dtype=get_shape_dtype(x),
        )
        return _mul_tracer(x_tracer, y)

    # any other array-like
    assert isinstance(x, get_args(AnyArrayLike)), f"Invalid input type for multiply: {x}"
    assert isinstance(y, get_args(AnyArrayLike)), f"Invalid input type for multiply: {y}"
    result = x * y
    assert isinstance(result, get_args(AnyArrayLike))
    return result


## Addition ###########################
def constraints_add_sub(
    x: mathopt.Variable | None,
    y: mathopt.Variable | None,
    out: Any | None,
) -> list[mathopt.BoundedLinearExpression]:
    if out is None:
        assert x is None and y is None
        return []
    assert isinstance(out, mathopt.LinearSum)
    constraints = []
    if x is not None:
        constraints.append(x == out)
    if y is not None:
        constraints.append(y == out)
    return constraints


def get_add_original():
    if hasattr(jax.numpy, "_orig_add"):
        return jax.numpy._orig_add
    return jax.numpy.add


@overload
def add(x: Unitful, y: Unitful) -> Unitful: ...


@overload
def add(x: AnyArrayLike, y: Unitful) -> Unitful: ...


@overload
def add(x: Unitful, y: AnyArrayLike) -> Unitful: ...


@overload
def add(x: int, y: int) -> int: ...


@overload
def add(x: int, y: float) -> float: ...


@overload
def add(x: float, y: int) -> float: ...


@overload
def add(x: float, y: float) -> float: ...


@overload
def add(x: complex, y: int | float) -> complex: ...


@overload
def add(x: int | float | complex, y: complex) -> complex: ...


@overload
def add(x: int | float | complex | np.number, y: np.ndarray) -> np.ndarray: ...


@overload
def add(x: np.ndarray, y: int | float | complex | np.number) -> np.ndarray: ...


@overload
def add(x: int | float | complex | np.number, y: jax.Array) -> jax.Array: ...


@overload
def add(x: jax.Array, y: int | float | complex | np.number) -> jax.Array: ...


@overload
def add(x: jax.Array, y: jax.Array) -> jax.Array: ...


@overload
def add(x: jax.Array, y: np.ndarray) -> jax.Array: ...


@overload
def add(x: np.ndarray, y: jax.Array) -> jax.Array: ...


@overload
def add(x: np.ndarray, y: np.ndarray) -> np.ndarray: ...


def add(x: AnyUnitType, y: AnyUnitType) -> AnyUnitType:
    def _add_unitful(x: Unitful, y: Unitful):
        if x.unit != y.unit:
            raise ValueError(f"Cannot add Unitful arrays with different units: {x.unit} vs {y.unit}")
        new_scale, fx, fy = handle_different_scales(x.scale, y.scale)
        new_val = x.val * fx + y.val * fy
        return Unitful(val=new_val, unit=x.unit, scale=new_scale, optimize_scale=False)

    if isinstance(x, Unitful) and isinstance(y, Unitful):
        return _add_unitful(x, y)
    elif isinstance(x, Unitful) and not isinstance(y, (Unitful | UnitfulTracer)):
        y_unitful = Unitful(val=y)
        return _add_unitful(x, y_unitful)
    elif not isinstance(x, (Unitful | UnitfulTracer)) and isinstance(y, Unitful):
        x_unitful = Unitful(val=x)
        return _add_unitful(x_unitful, y)

    # Tracer handling
    def _add_tracer(x: UnitfulTracer, y: UnitfulTracer):
        new_static_unitful = None
        if x.static_unitful is not None and y.static_unitful is not None:
            new_static_unitful = x.static_unitful + y.static_unitful
        result = UnitfulTracer(unit=x.unit, static_unitful=new_static_unitful)
        node = OperatorNode(
            op_name="add",
            op_kwargs={"x": x, "y": y},
            output=result,
        )
        register_node_full(node)
        return result

    if isinstance(x, UnitfulTracer) and isinstance(y, UnitfulTracer):
        return _add_tracer(x, y)
    if isinstance(x, UnitfulTracer):
        assert isinstance(y, StaticArrayLike | Unitful)
        y_unitful = Unitful(val=y) if not isinstance(y, Unitful) else y
        tracer_unit = None if not isinstance(y, Unitful) else y.unit
        y_tracer = UnitfulTracer(unit=tracer_unit, static_unitful=y_unitful, value=y)
        return _add_tracer(x, y_tracer)
    if isinstance(y, UnitfulTracer):
        assert isinstance(x, StaticArrayLike | Unitful)
        x_unitful = Unitful(val=x) if not isinstance(x, Unitful) else x
        tracer_unit = None if not isinstance(x, Unitful) else x.unit
        x_tracer = UnitfulTracer(unit=tracer_unit, static_unitful=x_unitful, value=x)
        return _add_tracer(x_tracer, y)

    # any other array-like
    assert isinstance(x, get_args(AnyArrayLike)), f"Invalid input type for add: {x}"
    assert isinstance(y, get_args(AnyArrayLike)), f"Invalid input type for add: {y}"
    result = x + y
    assert isinstance(result, get_args(AnyArrayLike))
    return result


## Subtraction ###########################


def get_subtract_original():
    if hasattr(jax.numpy, "_orig_subtract"):
        return jax.numpy._orig_subtract
    return jax.numpy.subtract


@overload
def subtract(x: Unitful, y: Unitful) -> Unitful: ...


@overload
def subtract(x: AnyArrayLike, y: Unitful) -> Unitful: ...


@overload
def subtract(x: Unitful, y: AnyArrayLike) -> Unitful: ...


@overload
def subtract(x: int, y: int) -> int: ...


@overload
def subtract(x: int, y: float) -> float: ...


@overload
def subtract(x: float, y: int) -> float: ...


@overload
def subtract(x: float, y: float) -> float: ...


@overload
def subtract(x: complex, y: int | float) -> complex: ...


@overload
def subtract(x: int | float | complex, y: complex) -> complex: ...


@overload
def subtract(x: int | float | complex | np.number, y: np.ndarray) -> np.ndarray: ...


@overload
def subtract(x: np.ndarray, y: int | float | complex | np.number) -> np.ndarray: ...


@overload
def subtract(x: int | float | complex | np.number, y: jax.Array) -> jax.Array: ...


@overload
def subtract(x: jax.Array, y: int | float | complex | np.number) -> jax.Array: ...


@overload
def subtract(x: jax.Array, y: jax.Array) -> jax.Array: ...


@overload
def subtract(x: jax.Array, y: np.ndarray) -> jax.Array: ...


@overload
def subtract(x: np.ndarray, y: jax.Array) -> jax.Array: ...


@overload
def subtract(x: np.ndarray, y: np.ndarray) -> np.ndarray: ...


def subtract(x: AnyUnitType, y: AnyUnitType) -> AnyUnitType:
    def _sub_unitful(x: Unitful, y: Unitful):
        if x.unit != y.unit:
            raise ValueError(f"Cannot subtract Unitful arrays with different units: {x.unit} vs {y.unit}")
        new_scale, fx, fy = handle_different_scales(x.scale, y.scale)
        new_val = x.val * fx - y.val * fy
        return Unitful(val=new_val, unit=x.unit, scale=new_scale, optimize_scale=False)

    if isinstance(x, Unitful) and isinstance(y, Unitful):
        return _sub_unitful(x, y)
    elif isinstance(x, Unitful) and not isinstance(y, (Unitful | UnitfulTracer)):
        y_unitful = Unitful(val=y)
        return _sub_unitful(x, y_unitful)
    elif not isinstance(x, (Unitful | UnitfulTracer)) and isinstance(y, Unitful):
        x_unitful = Unitful(val=x)
        return _sub_unitful(x_unitful, y)

    # Tracer handling
    def _sub_tracer(x: UnitfulTracer, y: UnitfulTracer):
        new_static_unitful = None
        if x.static_unitful is not None and y.static_unitful is not None:
            new_static_unitful = x.static_unitful - y.static_unitful
        result = UnitfulTracer(unit=x.unit, static_unitful=new_static_unitful)
        node = OperatorNode(
            op_name="subtract",
            op_kwargs={"x": x, "y": y},
            output=result,
        )
        register_node_full(node)
        return result

    if isinstance(x, UnitfulTracer) and isinstance(y, UnitfulTracer):
        return _sub_tracer(x, y)
    if isinstance(x, UnitfulTracer):
        assert isinstance(y, StaticArrayLike | Unitful)
        y_unitful = Unitful(val=y) if not isinstance(y, Unitful) else y
        tracer_unit = None if not isinstance(y, Unitful) else y.unit
        y_tracer = UnitfulTracer(unit=tracer_unit, static_unitful=y_unitful, value=y)
        return _sub_tracer(x, y_tracer)
    if isinstance(y, UnitfulTracer):
        assert isinstance(x, StaticArrayLike | Unitful)
        x_unitful = Unitful(val=x) if not isinstance(x, Unitful) else x
        tracer_unit = None if not isinstance(x, Unitful) else x.unit
        x_tracer = UnitfulTracer(unit=tracer_unit, static_unitful=x_unitful, value=x)
        return _sub_tracer(x_tracer, y)

    # any other array-like
    assert isinstance(x, get_args(AnyArrayLike)), f"Invalid input type for subtract: {x}"
    assert isinstance(y, get_args(AnyArrayLike)), f"Invalid input type for subtract: {y}"
    result = x - y
    assert isinstance(result, get_args(AnyArrayLike))
    return result


## Division ###########################
def constraints_divide(
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
        return [-y == out]
    if y is None:
        assert x is not None
        return [x == out]
    return [x - y == out]


def get_divide_original():
    if hasattr(jax.numpy, "_orig_divide"):
        return jax.numpy._orig_divide
    return jax.numpy.divide


@overload
def divide(x: Unitful, y: Unitful) -> Unitful: ...


@overload
def divide(x: AnyArrayLike, y: Unitful) -> Unitful: ...


@overload
def divide(x: Unitful, y: AnyArrayLike) -> Unitful: ...


@overload
def divide(x: int, y: int) -> int: ...


@overload
def divide(x: int, y: float) -> float: ...


@overload
def divide(x: float, y: int) -> float: ...


@overload
def divide(x: float, y: float) -> float: ...


@overload
def divide(x: complex, y: int | float) -> complex: ...


@overload
def divide(x: int | float | complex, y: complex) -> complex: ...


@overload
def divide(x: int | float | complex | np.number, y: np.ndarray) -> np.ndarray: ...


@overload
def divide(x: np.ndarray, y: int | float | complex | np.number) -> np.ndarray: ...


@overload
def divide(x: int | float | complex | np.number, y: jax.Array) -> jax.Array: ...


@overload
def divide(x: jax.Array, y: int | float | complex | np.number) -> jax.Array: ...


@overload
def divide(x: jax.Array, y: jax.Array) -> jax.Array: ...


@overload
def divide(x: jax.Array, y: np.ndarray) -> jax.Array: ...


@overload
def divide(x: np.ndarray, y: jax.Array) -> jax.Array: ...


@overload
def divide(x: np.ndarray, y: np.ndarray) -> np.ndarray: ...


def divide(x: AnyUnitType, y: AnyUnitType) -> AnyUnitType:
    def _div_unitful(x: Unitful, y: Unitful):
        unit_dict = {k: v for k, v in x.unit.items()}
        for k, v in y.unit.items():
            unit_dict[k] = unit_dict.get(k, 0) - v
            if unit_dict[k] == 0:
                del unit_dict[k]
        new_unit = Unit(unit_dict)
        new_val = x.val / y.val
        new_scale = x.scale - y.scale
        return Unitful(val=new_val, unit=new_unit, scale=new_scale, optimize_scale=False)

    if isinstance(x, Unitful) and isinstance(y, Unitful):
        return _div_unitful(x, y)
    elif isinstance(x, Unitful) and not isinstance(y, (Unitful | UnitfulTracer)):
        y_unitful = Unitful(val=y)
        return _div_unitful(x, y_unitful)
    elif not isinstance(x, (Unitful | UnitfulTracer)) and isinstance(y, Unitful):
        x_unitful = Unitful(val=x)
        return _div_unitful(x_unitful, y)

    # Tracer handling
    def _div_tracer(x: UnitfulTracer, y: UnitfulTracer):
        new_unit = None
        if x.unit is not None and y.unit is not None:
            unit_dict = {k: v for k, v in x.unit.items()}
            for k, v in y.unit.items():
                unit_dict[k] = unit_dict.get(k, 0) - v
                if unit_dict[k] == 0:
                    del unit_dict[k]
            new_unit = Unit(unit_dict)
        elif x.unit is not None:
            new_unit = x.unit
        new_static_unitful = None
        if x.static_unitful is not None and y.static_unitful is not None:
            new_static_unitful = x.static_unitful / y.static_unitful
        result = UnitfulTracer(unit=new_unit, static_unitful=new_static_unitful)
        node = OperatorNode(
            op_name="divide",
            op_kwargs={"x": x, "y": y},
            output=result,
        )
        register_node_full(node)
        return result

    if isinstance(x, UnitfulTracer) and isinstance(y, UnitfulTracer):
        return _div_tracer(x, y)
    if isinstance(x, UnitfulTracer):
        assert isinstance(y, StaticArrayLike | Unitful)
        y_unitful = Unitful(val=y) if not isinstance(y, Unitful) else y
        tracer_unit = None if not isinstance(y, Unitful) else y.unit
        y_tracer = UnitfulTracer(unit=tracer_unit, static_unitful=y_unitful, value=y)
        return _div_tracer(x, y_tracer)
    if isinstance(y, UnitfulTracer):
        assert isinstance(x, StaticArrayLike | Unitful)
        x_unitful = Unitful(val=x) if not isinstance(x, Unitful) else x
        tracer_unit = None if not isinstance(x, Unitful) else x.unit
        x_tracer = UnitfulTracer(unit=tracer_unit, static_unitful=x_unitful, value=x)
        return _div_tracer(x_tracer, y)

    # any other array-like
    assert isinstance(x, get_args(AnyArrayLike)), f"Invalid input type for divide: {x}"
    assert isinstance(y, get_args(AnyArrayLike)), f"Invalid input type for divide: {y}"
    result = x / y
    assert isinstance(result, get_args(AnyArrayLike))
    return result
