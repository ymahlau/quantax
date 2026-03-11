from __future__ import annotations

from typing import Any, overload

import jax
import numpy as np
from ortools.math_opt.python import mathopt

from quantax.core.typing import AnyArrayLike
from quantax.core.unit import EMPTY_UNIT
from quantax.functional.utils import binary_op_from_func
from quantax.tracing.glob import OperatorNode, register_node_full
from quantax.tracing.tracer import UnitfulTracer
from quantax.tracing.types import AnyUnitType
from quantax.unitful.alignment import align_scales
from quantax.unitful.unitful import Unitful


def constraints_comparison(
    x: mathopt.Variable | None,
    y: mathopt.Variable | None,
    out: Any | None,
) -> list[mathopt.BoundedLinearExpression]:
    assert isinstance(out, mathopt.LinearSum)
    assert x is not None and y is not None
    return [x == y, out == 0]  # ty:ignore[invalid-return-type]


## Equal ###########################
def get_eq_original():
    if hasattr(jax.numpy, "_orig_equal"):
        return jax.numpy._orig_equal
    return jax.numpy.equal


@overload
def eq(x: Unitful, y: Unitful) -> Unitful: ...


@overload
def eq(x: AnyArrayLike, y: Unitful) -> Unitful: ...


@overload
def eq(x: Unitful, y: AnyArrayLike) -> Unitful: ...


@overload
def eq(x: int, y: int) -> bool: ...


@overload
def eq(x: int, y: float) -> bool: ...


@overload
def eq(x: float, y: int) -> bool: ...


@overload
def eq(x: float, y: float) -> bool: ...


@overload
def eq(x: complex, y: int | float) -> bool: ...


@overload
def eq(x: int | float | complex, y: complex) -> bool: ...


@overload
def eq(x: int | float | complex | np.number, y: np.ndarray) -> np.ndarray: ...


@overload
def eq(x: np.ndarray, y: int | float | complex | np.number) -> np.ndarray: ...


@overload
def eq(x: int | float | complex | np.number, y: jax.Array) -> jax.Array: ...


@overload
def eq(x: jax.Array, y: int | float | complex | np.number) -> jax.Array: ...


@overload
def eq(x: jax.Array, y: jax.Array) -> jax.Array: ...


@overload
def eq(x: jax.Array, y: np.ndarray) -> jax.Array: ...


@overload
def eq(x: np.ndarray, y: jax.Array) -> jax.Array: ...


@overload
def eq(x: np.ndarray, y: np.ndarray) -> np.ndarray: ...


def eq(x: AnyUnitType, y: AnyUnitType) -> AnyUnitType:
    # Unitful handling
    def _eq_unitful(x: Unitful, y: Unitful):
        x_align, y_align = align_scales(x, y)
        new_val = x_align.val == y_align.val
        # Equality results in a dimensionless boolean
        return Unitful(val=new_val)

    # Tracer handling
    def _eq_tracer(x: UnitfulTracer, y: UnitfulTracer):
        assert x.unit == y.unit, f"Unit mismatch for equality: {x.unit} vs {y.unit}"
        new_static_unitful = None
        if x.static_unitful is not None and y.static_unitful is not None:
            new_static_unitful = _eq_unitful(x.static_unitful, y.static_unitful)

        # Output tracer is dimensionless
        result = UnitfulTracer(unit=EMPTY_UNIT, static_unitful=new_static_unitful)
        node = OperatorNode(
            op_name="eq",
            op_kwargs={"x": x, "y": y},
            output=result,
        )
        register_node_full(node)
        return result

    result = binary_op_from_func(
        unitful_handler=_eq_unitful,
        tracer_handler=_eq_tracer,
        standard_handler=lambda x, y: x == y,
        x=x,
        y=y,
    )
    return result


## Not Equal ###########################
def get_ne_original():
    if hasattr(jax.numpy, "_orig_not_equal"):
        return jax.numpy._orig_not_equal
    return jax.numpy.not_equal


@overload
def ne(x: Unitful, y: Unitful) -> bool | jax.Array | np.ndarray: ...


@overload
def ne(x: AnyArrayLike, y: Unitful) -> bool | jax.Array | np.ndarray: ...


@overload
def ne(x: Unitful, y: AnyArrayLike) -> bool | jax.Array | np.ndarray: ...


@overload
def ne(x: int, y: int) -> bool: ...


@overload
def ne(x: int, y: float) -> bool: ...


@overload
def ne(x: float, y: int) -> bool: ...


@overload
def ne(x: float, y: float) -> bool: ...


@overload
def ne(x: complex, y: int | float) -> bool: ...


@overload
def ne(x: int | float | complex, y: complex) -> bool: ...


@overload
def ne(x: int | float | complex | np.number, y: np.ndarray) -> np.ndarray: ...


@overload
def ne(x: np.ndarray, y: int | float | complex | np.number) -> np.ndarray: ...


@overload
def ne(x: int | float | complex | np.number, y: jax.Array) -> jax.Array: ...


@overload
def ne(x: jax.Array, y: int | float | complex | np.number) -> jax.Array: ...


@overload
def ne(x: jax.Array, y: jax.Array) -> jax.Array: ...


@overload
def ne(x: jax.Array, y: np.ndarray) -> jax.Array: ...


@overload
def ne(x: np.ndarray, y: jax.Array) -> jax.Array: ...


@overload
def ne(x: np.ndarray, y: np.ndarray) -> np.ndarray: ...


def ne(x: AnyUnitType, y: AnyUnitType) -> AnyUnitType:
    # Unitful handling
    def _ne_unitful(x: Unitful, y: Unitful):
        x_align, y_align = align_scales(x, y)
        # Note: using the aligned values instead of the raw x.val != y.val
        new_val = x_align.val != y_align.val
        # Inequality results in a dimensionless boolean
        return Unitful(val=new_val)

    # Tracer handling
    def _ne_tracer(x: UnitfulTracer, y: UnitfulTracer):
        assert x.unit == y.unit, f"Unit mismatch for inequality: {x.unit} vs {y.unit}"
        new_static_unitful = None
        if x.static_unitful is not None and y.static_unitful is not None:
            new_static_unitful = _ne_unitful(x.static_unitful, y.static_unitful)

        # Output tracer is dimensionless
        result = UnitfulTracer(unit=EMPTY_UNIT, static_unitful=new_static_unitful)
        node = OperatorNode(
            op_name="neq",
            op_kwargs={"x": x, "y": y},
            output=result,
        )
        register_node_full(node)
        return result

    result = binary_op_from_func(
        unitful_handler=_ne_unitful,
        tracer_handler=_ne_tracer,
        standard_handler=lambda x, y: x != y,
        x=x,
        y=y,
    )
    return result


## Less Than ###########################
def get_lt_original():
    if hasattr(jax.numpy, "_orig_less"):
        return jax.numpy._orig_less
    return jax.numpy.less


@overload
def lt(x: Unitful, y: Unitful) -> bool | jax.Array | np.ndarray: ...


@overload
def lt(x: AnyArrayLike, y: Unitful) -> bool | jax.Array | np.ndarray: ...


@overload
def lt(x: Unitful, y: AnyArrayLike) -> bool | jax.Array | np.ndarray: ...


@overload
def lt(x: int, y: int) -> bool: ...


@overload
def lt(x: int, y: float) -> bool: ...


@overload
def lt(x: float, y: int) -> bool: ...


@overload
def lt(x: float, y: float) -> bool: ...


@overload
def lt(x: int | float | complex | np.number, y: np.ndarray) -> np.ndarray: ...


@overload
def lt(x: np.ndarray, y: int | float | complex | np.number) -> np.ndarray: ...


@overload
def lt(x: int | float | complex | np.number, y: jax.Array) -> jax.Array: ...


@overload
def lt(x: jax.Array, y: int | float | complex | np.number) -> jax.Array: ...


@overload
def lt(x: jax.Array, y: jax.Array) -> jax.Array: ...


@overload
def lt(x: jax.Array, y: np.ndarray) -> jax.Array: ...


@overload
def lt(x: np.ndarray, y: jax.Array) -> jax.Array: ...


@overload
def lt(x: np.ndarray, y: np.ndarray) -> np.ndarray: ...


def lt(x: AnyUnitType, y: AnyUnitType) -> AnyUnitType:
    # Unitful handling
    def _lt_unitful(x: Unitful, y: Unitful):
        x_align, y_align = align_scales(x, y)
        x_val, y_val = x_align.val, y_align.val
        assert not isinstance(x_val, complex), f"Cannot compare < with complex values: {x_val}"
        assert not isinstance(y_val, complex), f"Cannot compare < with complex values: {y_val}"
        new_val = x_val < y_val
        # Less than results in a dimensionless boolean
        return Unitful(val=new_val)

    # Tracer handling
    def _lt_tracer(x: UnitfulTracer, y: UnitfulTracer):
        assert x.unit == y.unit, f"Unit mismatch for less than: {x.unit} vs {y.unit}"
        new_static_unitful = None
        if x.static_unitful is not None and y.static_unitful is not None:
            new_static_unitful = _lt_unitful(x.static_unitful, y.static_unitful)

        # Output tracer is dimensionless
        result = UnitfulTracer(unit=EMPTY_UNIT, static_unitful=new_static_unitful)
        node = OperatorNode(
            op_name="lt",
            op_kwargs={"x": x, "y": y},
            output=result,
        )
        register_node_full(node)
        return result

    result = binary_op_from_func(
        unitful_handler=_lt_unitful,
        tracer_handler=_lt_tracer,
        standard_handler=lambda x, y: x < y,
        x=x,
        y=y,
    )
    return result


## Less Equal ###########################
def get_le_original():
    if hasattr(jax.numpy, "_orig_less_equal"):
        return jax.numpy._orig_less_equal
    return jax.numpy.less_equal


@overload
def le(x: Unitful, y: Unitful) -> bool | jax.Array | np.ndarray: ...


@overload
def le(x: AnyArrayLike, y: Unitful) -> bool | jax.Array | np.ndarray: ...


@overload
def le(x: Unitful, y: AnyArrayLike) -> bool | jax.Array | np.ndarray: ...


@overload
def le(x: int, y: int) -> bool: ...


@overload
def le(x: int, y: float) -> bool: ...


@overload
def le(x: float, y: int) -> bool: ...


@overload
def le(x: float, y: float) -> bool: ...


@overload
def le(x: int | float | complex | np.number, y: np.ndarray) -> np.ndarray: ...


@overload
def le(x: np.ndarray, y: int | float | complex | np.number) -> np.ndarray: ...


@overload
def le(x: int | float | complex | np.number, y: jax.Array) -> jax.Array: ...


@overload
def le(x: jax.Array, y: int | float | complex | np.number) -> jax.Array: ...


@overload
def le(x: jax.Array, y: jax.Array) -> jax.Array: ...


@overload
def le(x: jax.Array, y: np.ndarray) -> jax.Array: ...


@overload
def le(x: np.ndarray, y: jax.Array) -> jax.Array: ...


@overload
def le(x: np.ndarray, y: np.ndarray) -> np.ndarray: ...


def le(x: AnyUnitType, y: AnyUnitType) -> AnyUnitType:
    # Unitful handling
    def _le_unitful(x: Unitful, y: Unitful):
        x_align, y_align = align_scales(x, y)
        x_val, y_val = x_align.val, y_align.val
        assert not isinstance(x_val, complex), f"Cannot compare <= with complex values: {x_val}"
        assert not isinstance(y_val, complex), f"Cannot compare <= with complex values: {y_val}"
        new_val = x_val <= y_val
        # Comparison results in a dimensionless boolean
        return Unitful(val=new_val)

    # Tracer handling
    def _le_tracer(x: UnitfulTracer, y: UnitfulTracer):
        assert x.unit == y.unit, f"Unit mismatch for less than or equal: {x.unit} vs {y.unit}"
        new_static_unitful = None
        if x.static_unitful is not None and y.static_unitful is not None:
            new_static_unitful = _le_unitful(x.static_unitful, y.static_unitful)

        # Output tracer is dimensionless
        result = UnitfulTracer(unit=EMPTY_UNIT, static_unitful=new_static_unitful)
        node = OperatorNode(
            op_name="le",
            op_kwargs={"x": x, "y": y},
            output=result,
        )
        register_node_full(node)
        return result

    result = binary_op_from_func(
        unitful_handler=_le_unitful,
        tracer_handler=_le_tracer,
        standard_handler=lambda x, y: x <= y,
        x=x,
        y=y,
    )
    return result


## Greater Than ###########################
def get_gt_original():
    if hasattr(jax.numpy, "_orig_greater"):
        return jax.numpy._orig_greater
    return jax.numpy.greater


@overload
def gt(x: Unitful, y: Unitful) -> bool | jax.Array | np.ndarray: ...


@overload
def gt(x: AnyArrayLike, y: Unitful) -> bool | jax.Array | np.ndarray: ...


@overload
def gt(x: Unitful, y: AnyArrayLike) -> bool | jax.Array | np.ndarray: ...


@overload
def gt(x: int, y: int) -> bool: ...


@overload
def gt(x: int, y: float) -> bool: ...


@overload
def gt(x: float, y: int) -> bool: ...


@overload
def gt(x: float, y: float) -> bool: ...


@overload
def gt(x: int | float | np.number, y: np.ndarray) -> np.ndarray: ...


@overload
def gt(x: np.ndarray, y: int | float | np.number) -> np.ndarray: ...


@overload
def gt(x: int | float | np.number, y: jax.Array) -> jax.Array: ...


@overload
def gt(x: jax.Array, y: int | float | np.number) -> jax.Array: ...


@overload
def gt(x: jax.Array, y: jax.Array) -> jax.Array: ...


@overload
def gt(x: jax.Array, y: np.ndarray) -> jax.Array: ...


@overload
def gt(x: np.ndarray, y: jax.Array) -> jax.Array: ...


@overload
def gt(x: np.ndarray, y: np.ndarray) -> np.ndarray: ...


def gt(x: AnyUnitType, y: AnyUnitType) -> AnyUnitType:
    # Unitful handling
    def _gt_unitful(x: Unitful, y: Unitful):
        x_align, y_align = align_scales(x, y)
        x_val, y_val = x_align.val, y_align.val
        assert not isinstance(x_val, complex), f"Cannot compare > with complex values: {x_val}"
        assert not isinstance(y_val, complex), f"Cannot compare > with complex values: {y_val}"
        new_val = x_val > y_val
        # Greater than results in a dimensionless boolean
        return Unitful(val=new_val)

    # Tracer handling
    def _gt_tracer(x: UnitfulTracer, y: UnitfulTracer):
        assert x.unit == y.unit, f"Unit mismatch for greater than: {x.unit} vs {y.unit}"
        new_static_unitful = None
        if x.static_unitful is not None and y.static_unitful is not None:
            new_static_unitful = _gt_unitful(x.static_unitful, y.static_unitful)

        # Output tracer is dimensionless
        result = UnitfulTracer(unit=EMPTY_UNIT, static_unitful=new_static_unitful)
        node = OperatorNode(
            op_name="gt",
            op_kwargs={"x": x, "y": y},
            output=result,
        )
        register_node_full(node)
        return result

    result = binary_op_from_func(
        unitful_handler=_gt_unitful,
        tracer_handler=_gt_tracer,
        standard_handler=lambda x, y: x > y,
        x=x,
        y=y,
    )
    return result


## Greater Equal ###########################
def get_ge_original():
    if hasattr(jax.numpy, "_orig_greater_equal"):
        return jax.numpy._orig_greater_equal
    return jax.numpy.greater_equal


@overload
def ge(x: Unitful, y: Unitful) -> bool | jax.Array | np.ndarray: ...


@overload
def ge(x: AnyArrayLike, y: Unitful) -> bool | jax.Array | np.ndarray: ...


@overload
def ge(x: Unitful, y: AnyArrayLike) -> bool | jax.Array | np.ndarray: ...


@overload
def ge(x: int, y: int) -> bool: ...


@overload
def ge(x: int, y: float) -> bool: ...


@overload
def ge(x: float, y: int) -> bool: ...


@overload
def ge(x: float, y: float) -> bool: ...


@overload
def ge(x: int | float | complex | np.number, y: np.ndarray) -> np.ndarray: ...


@overload
def ge(x: np.ndarray, y: int | float | complex | np.number) -> np.ndarray: ...


@overload
def ge(x: int | float | complex | np.number, y: jax.Array) -> jax.Array: ...


@overload
def ge(x: jax.Array, y: int | float | complex | np.number) -> jax.Array: ...


@overload
def ge(x: jax.Array, y: jax.Array) -> jax.Array: ...


@overload
def ge(x: jax.Array, y: np.ndarray) -> jax.Array: ...


@overload
def ge(x: np.ndarray, y: jax.Array) -> jax.Array: ...


@overload
def ge(x: np.ndarray, y: np.ndarray) -> np.ndarray: ...


def ge(x: AnyUnitType, y: AnyUnitType) -> AnyUnitType:
    # Unitful handling
    def _ge_unitful(x: Unitful, y: Unitful):
        x_align, y_align = align_scales(x, y)
        x_val, y_val = x_align.val, y_align.val
        assert not isinstance(x_val, complex), f"Cannot compare >= with complex values: {x_val}"
        assert not isinstance(y_val, complex), f"Cannot compare >= with complex values: {y_val}"
        new_val = x_val >= y_val
        # Greater than or equal to results in a dimensionless boolean
        return Unitful(val=new_val)

    # Tracer handling
    def _ge_tracer(x: UnitfulTracer, y: UnitfulTracer):
        assert x.unit == y.unit, f"Unit mismatch for greater than or equal to: {x.unit} vs {y.unit}"
        new_static_unitful = None
        if x.static_unitful is not None and y.static_unitful is not None:
            new_static_unitful = _ge_unitful(x.static_unitful, y.static_unitful)

        # Output tracer is dimensionless
        result = UnitfulTracer(unit=EMPTY_UNIT, static_unitful=new_static_unitful)
        node = OperatorNode(
            op_name="ge",
            op_kwargs={"x": x, "y": y},
            output=result,
        )
        register_node_full(node)
        return result

    result = binary_op_from_func(
        unitful_handler=_ge_unitful,
        tracer_handler=_ge_tracer,
        standard_handler=lambda x, y: x >= y,
        x=x,
        y=y,
    )
    return result


# Array equal
def get_array_equal_original():
    if hasattr(jax.numpy, "_orig_array_equal"):
        return jax.numpy._orig_array_equal
    return jax.numpy.array_equal


@overload
def array_equal(x: Unitful, y: Unitful) -> Unitful: ...


@overload
def array_equal(x: AnyArrayLike, y: Unitful) -> Unitful: ...


@overload
def array_equal(x: Unitful, y: AnyArrayLike) -> Unitful: ...


@overload
def array_equal(x: jax.Array, y: jax.Array | np.ndarray) -> jax.Array: ...


@overload
def array_equal(x: jax.Array | np.ndarray, y: np.ndarray) -> jax.Array: ...


@overload
def array_equal(x: np.ndarray | int | float | complex, y: np.ndarray | int | float | complex) -> np.ndarray: ...


def array_equal(x: AnyUnitType, y: AnyUnitType) -> AnyUnitType:
    # Unitful handling
    def _array_equal_unitful(x: Unitful, y: Unitful):
        x_align, y_align = align_scales(x, y)
        orig_fn = get_array_equal_original()
        new_val = orig_fn(x_align.val, y_align.val)
        # array_equal results in a dimensionless boolean scalar
        return Unitful(val=new_val)

    # Tracer handling
    def _array_equal_tracer(x: UnitfulTracer, y: UnitfulTracer):
        assert x.unit == y.unit, f"Unit mismatch for array_equal: {x.unit} vs {y.unit}"
        new_static_unitful = None

        if x.static_unitful is not None and y.static_unitful is not None:
            # Recursively call array_equal to compute the static value
            new_static_unitful = array_equal(x.static_unitful, y.static_unitful)

        # Output tracer is a dimensionless scalar
        result = UnitfulTracer(unit=EMPTY_UNIT, static_unitful=new_static_unitful)
        node = OperatorNode(
            op_name="array_equal",
            op_kwargs={"x": x, "y": y},
            output=result,
        )
        register_node_full(node)
        return result

    def _array_equal_standard(x: AnyArrayLike, y: AnyArrayLike) -> AnyArrayLike:
        if isinstance(x, jax.Array) or isinstance(y, jax.Array):
            orig_fn = get_array_equal_original()
        else:
            orig_fn = np.array_equal
        return orig_fn(x, y)

    result = binary_op_from_func(
        unitful_handler=_array_equal_unitful,
        tracer_handler=_array_equal_tracer,
        standard_handler=_array_equal_standard,
        x=x,
        y=y,
    )
    return result
