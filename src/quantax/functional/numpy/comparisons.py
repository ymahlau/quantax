from __future__ import annotations
from quantax.tracing.tracer import UnitfulTracer
from quantax.unitful.alignment import align_scales
from quantax.tracing.glob import OperatorNode, register_node_full
from quantax.tracing.types import AnyUnitType

from typing import get_args, overload, Any
from ortools.math_opt.python import mathopt

import jax
import numpy as np

from quantax.core.typing import AnyArrayLike, StaticArrayLike
from quantax.core.utils import handle_different_scales
from quantax.unitful.unitful import Unitful


def constraints_comparison(
    x: mathopt.Variable | None,
    y: mathopt.Variable | None,
    out: Any | None,
) -> list[mathopt.BoundedLinearExpression]:
    assert out is None
    assert x is not None and y is not None
    return [x == y]  # ty:ignore[invalid-return-type]


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

    if isinstance(x, Unitful) and isinstance(y, Unitful):
        return _eq_unitful(x, y)
    elif isinstance(x, Unitful) and not isinstance(y, (Unitful, UnitfulTracer)):
        y_unitful = Unitful(val=y)
        return _eq_unitful(x, y_unitful)
    elif not isinstance(x, (Unitful, UnitfulTracer)) and isinstance(y, Unitful):
        x_unitful = Unitful(val=x)
        return _eq_unitful(x_unitful, y)

    # Tracer handling
    def _eq_tracer(x: UnitfulTracer, y: UnitfulTracer):
        assert x.unit == y.unit, f"Unit mismatch for equality: {x.unit} vs {y.unit}"
        new_static_unitful = None
        if x.static_unitful is not None and y.static_unitful is not None:
            new_static_unitful = x.static_unitful == y.static_unitful
        
        # Output tracer is dimensionless
        result = UnitfulTracer(unit=None, static_unitful=new_static_unitful)
        node = OperatorNode(
            op_name="eq",
            op_kwargs={"x": x, "y": y},
            output=result,
        )
        register_node_full(node)
        return result

    if isinstance(x, UnitfulTracer) and isinstance(y, UnitfulTracer):
        return _eq_tracer(x, y)
    if isinstance(x, UnitfulTracer):
        assert isinstance(y, (StaticArrayLike, Unitful))
        y_unitful = Unitful(val=y) if not isinstance(y, Unitful) else y
        tracer_unit = None if not isinstance(y, Unitful) else y.unit
        y_tracer = UnitfulTracer(unit=tracer_unit, static_unitful=y_unitful, value=y)
        return _eq_tracer(x, y_tracer)
    if isinstance(y, UnitfulTracer):
        assert isinstance(x, (StaticArrayLike, Unitful))
        x_unitful = Unitful(val=x) if not isinstance(x, Unitful) else x
        tracer_unit = None if not isinstance(x, Unitful) else x.unit
        x_tracer = UnitfulTracer(unit=tracer_unit, static_unitful=x_unitful, value=x)
        return _eq_tracer(x_tracer, y)

    # any other array-like
    assert isinstance(x, get_args(AnyArrayLike)), f"Invalid input type for equals: {x}"
    assert isinstance(y, get_args(AnyArrayLike)), f"Invalid input type for equals: {y}"
    result = x == y
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
    def _neq_unitful(x: Unitful, y: Unitful):
        x_align, y_align = align_scales(x, y)
        # Note: using the aligned values instead of the raw x.val != y.val
        new_val = x_align.val != y_align.val
        # Inequality results in a dimensionless boolean
        return Unitful(val=new_val)

    if isinstance(x, Unitful) and isinstance(y, Unitful):
        return _neq_unitful(x, y)
    elif isinstance(x, Unitful) and not isinstance(y, (Unitful, UnitfulTracer)):
        y_unitful = Unitful(val=y)
        return _neq_unitful(x, y_unitful)
    elif not isinstance(x, (Unitful, UnitfulTracer)) and isinstance(y, Unitful):
        x_unitful = Unitful(val=x)
        return _neq_unitful(x_unitful, y)

    # Tracer handling
    def _neq_tracer(x: UnitfulTracer, y: UnitfulTracer):
        assert x.unit == y.unit, f"Unit mismatch for inequality: {x.unit} vs {y.unit}"
        new_static_unitful = None
        if x.static_unitful is not None and y.static_unitful is not None:
            new_static_unitful = x.static_unitful != y.static_unitful
        
        # Output tracer is dimensionless
        result = UnitfulTracer(unit=None, static_unitful=new_static_unitful)
        node = OperatorNode(
            op_name="neq",
            op_kwargs={"x": x, "y": y},
            output=result,
        )
        register_node_full(node)
        return result

    if isinstance(x, UnitfulTracer) and isinstance(y, UnitfulTracer):
        return _neq_tracer(x, y)
    if isinstance(x, UnitfulTracer):
        assert isinstance(y, (StaticArrayLike, Unitful))
        y_unitful = Unitful(val=y) if not isinstance(y, Unitful) else y
        tracer_unit = None if not isinstance(y, Unitful) else y.unit
        y_tracer = UnitfulTracer(unit=tracer_unit, static_unitful=y_unitful, value=y)
        return _neq_tracer(x, y_tracer)
    if isinstance(y, UnitfulTracer):
        assert isinstance(x, (StaticArrayLike, Unitful))
        x_unitful = Unitful(val=x) if not isinstance(x, Unitful) else x
        tracer_unit = None if not isinstance(x, Unitful) else x.unit
        x_tracer = UnitfulTracer(unit=tracer_unit, static_unitful=x_unitful, value=x)
        return _neq_tracer(x_tracer, y)

    # any other array-like
    assert isinstance(x, get_args(AnyArrayLike)), f"Invalid input type for not equals: {x}"
    assert isinstance(y, get_args(AnyArrayLike)), f"Invalid input type for not equals: {y}"
    result = x != y
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

    if isinstance(x, Unitful) and isinstance(y, Unitful):
        return _lt_unitful(x, y)
    elif isinstance(x, Unitful) and not isinstance(y, (Unitful, UnitfulTracer)):
        y_unitful = Unitful(val=y)
        return _lt_unitful(x, y_unitful)
    elif not isinstance(x, (Unitful, UnitfulTracer)) and isinstance(y, Unitful):
        x_unitful = Unitful(val=x)
        return _lt_unitful(x_unitful, y)

    # Tracer handling
    def _lt_tracer(x: UnitfulTracer, y: UnitfulTracer):
        assert x.unit == y.unit, f"Unit mismatch for less than: {x.unit} vs {y.unit}"
        new_static_unitful = None
        if x.static_unitful is not None and y.static_unitful is not None:
            new_static_unitful = x.static_unitful < y.static_unitful
        
        # Output tracer is dimensionless
        result = UnitfulTracer(unit=None, static_unitful=new_static_unitful)
        node = OperatorNode(
            op_name="lt",
            op_kwargs={"x": x, "y": y},
            output=result,
        )
        register_node_full(node)
        return result

    if isinstance(x, UnitfulTracer) and isinstance(y, UnitfulTracer):
        return _lt_tracer(x, y)
    if isinstance(x, UnitfulTracer):
        assert isinstance(y, (StaticArrayLike, Unitful))
        y_unitful = Unitful(val=y) if not isinstance(y, Unitful) else y
        tracer_unit = None if not isinstance(y, Unitful) else y.unit
        y_tracer = UnitfulTracer(unit=tracer_unit, static_unitful=y_unitful, value=y)
        return _lt_tracer(x, y_tracer)
    if isinstance(y, UnitfulTracer):
        assert isinstance(x, (StaticArrayLike, Unitful))
        x_unitful = Unitful(val=x) if not isinstance(x, Unitful) else x
        tracer_unit = None if not isinstance(x, Unitful) else x.unit
        x_tracer = UnitfulTracer(unit=tracer_unit, static_unitful=x_unitful, value=x)
        return _lt_tracer(x_tracer, y)

    # any other array-like
    assert isinstance(x, get_args(AnyArrayLike)), f"Invalid input type for less than: {x}"
    assert isinstance(y, get_args(AnyArrayLike)), f"Invalid input type for less than: {y}"
    result = x < y
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

    if isinstance(x, Unitful) and isinstance(y, Unitful):
        return _le_unitful(x, y)
    elif isinstance(x, Unitful) and not isinstance(y, (Unitful, UnitfulTracer)):
        y_unitful = Unitful(val=y)
        return _le_unitful(x, y_unitful)
    elif not isinstance(x, (Unitful, UnitfulTracer)) and isinstance(y, Unitful):
        x_unitful = Unitful(val=x)
        return _le_unitful(x_unitful, y)

    # Tracer handling
    def _le_tracer(x: UnitfulTracer, y: UnitfulTracer):
        assert x.unit == y.unit, f"Unit mismatch for less than or equal: {x.unit} vs {y.unit}"
        new_static_unitful = None
        if x.static_unitful is not None and y.static_unitful is not None:
            new_static_unitful = x.static_unitful <= y.static_unitful
        
        # Output tracer is dimensionless
        result = UnitfulTracer(unit=None, static_unitful=new_static_unitful)
        node = OperatorNode(
            op_name="le",
            op_kwargs={"x": x, "y": y},
            output=result,
        )
        register_node_full(node)
        return result

    if isinstance(x, UnitfulTracer) and isinstance(y, UnitfulTracer):
        return _le_tracer(x, y)
    if isinstance(x, UnitfulTracer):
        assert isinstance(y, (StaticArrayLike, Unitful))
        y_unitful = Unitful(val=y) if not isinstance(y, Unitful) else y
        tracer_unit = None if not isinstance(y, Unitful) else y.unit
        y_tracer = UnitfulTracer(unit=tracer_unit, static_unitful=y_unitful, value=y)
        return _le_tracer(x, y_tracer)
    if isinstance(y, UnitfulTracer):
        assert isinstance(x, (StaticArrayLike, Unitful))
        x_unitful = Unitful(val=x) if not isinstance(x, Unitful) else x
        tracer_unit = None if not isinstance(x, Unitful) else x.unit
        x_tracer = UnitfulTracer(unit=tracer_unit, static_unitful=x_unitful, value=x)
        return _le_tracer(x_tracer, y)

    # any other array-like
    assert isinstance(x, get_args(AnyArrayLike)), f"Invalid input type for less than or equal: {x}"
    assert isinstance(y, get_args(AnyArrayLike)), f"Invalid input type for less than or equal: {y}"
    result = x <= y
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

    if isinstance(x, Unitful) and isinstance(y, Unitful):
        return _gt_unitful(x, y)
    elif isinstance(x, Unitful) and not isinstance(y, (Unitful, UnitfulTracer)):
        y_unitful = Unitful(val=y)
        return _gt_unitful(x, y_unitful)
    elif not isinstance(x, (Unitful, UnitfulTracer)) and isinstance(y, Unitful):
        x_unitful = Unitful(val=x)
        return _gt_unitful(x_unitful, y)

    # Tracer handling
    def _gt_tracer(x: UnitfulTracer, y: UnitfulTracer):
        assert x.unit == y.unit, f"Unit mismatch for greater than: {x.unit} vs {y.unit}"
        new_static_unitful = None
        if x.static_unitful is not None and y.static_unitful is not None:
            new_static_unitful = x.static_unitful > y.static_unitful
        
        # Output tracer is dimensionless
        result = UnitfulTracer(unit=None, static_unitful=new_static_unitful)
        node = OperatorNode(
            op_name="gt",
            op_kwargs={"x": x, "y": y},
            output=result,
        )
        register_node_full(node)
        return result

    if isinstance(x, UnitfulTracer) and isinstance(y, UnitfulTracer):
        return _gt_tracer(x, y)
    if isinstance(x, UnitfulTracer):
        assert isinstance(y, (StaticArrayLike, Unitful))
        y_unitful = Unitful(val=y) if not isinstance(y, Unitful) else y
        tracer_unit = None if not isinstance(y, Unitful) else y.unit
        y_tracer = UnitfulTracer(unit=tracer_unit, static_unitful=y_unitful, value=y)
        return _gt_tracer(x, y_tracer)
    if isinstance(y, UnitfulTracer):
        assert isinstance(x, (StaticArrayLike, Unitful))
        x_unitful = Unitful(val=x) if not isinstance(x, Unitful) else x
        tracer_unit = None if not isinstance(x, Unitful) else x.unit
        x_tracer = UnitfulTracer(unit=tracer_unit, static_unitful=x_unitful, value=x)
        return _gt_tracer(x_tracer, y)

    # any other array-like
    assert isinstance(x, get_args(AnyArrayLike)), f"Invalid input type for greater than: {x}"
    assert isinstance(y, get_args(AnyArrayLike)), f"Invalid input type for greater than: {y}"
    result = x > y
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

    if isinstance(x, Unitful) and isinstance(y, Unitful):
        return _ge_unitful(x, y)
    elif isinstance(x, Unitful) and not isinstance(y, (Unitful, UnitfulTracer)):
        y_unitful = Unitful(val=y)
        return _ge_unitful(x, y_unitful)
    elif not isinstance(x, (Unitful, UnitfulTracer)) and isinstance(y, Unitful):
        x_unitful = Unitful(val=x)
        return _ge_unitful(x_unitful, y)

    # Tracer handling
    def _ge_tracer(x: UnitfulTracer, y: UnitfulTracer):
        assert x.unit == y.unit, f"Unit mismatch for greater than or equal to: {x.unit} vs {y.unit}"
        new_static_unitful = None
        if x.static_unitful is not None and y.static_unitful is not None:
            new_static_unitful = x.static_unitful >= y.static_unitful
        
        # Output tracer is dimensionless
        result = UnitfulTracer(unit=None, static_unitful=new_static_unitful)
        node = OperatorNode(
            op_name="ge",
            op_kwargs={"x": x, "y": y},
            output=result,
        )
        register_node_full(node)
        return result

    if isinstance(x, UnitfulTracer) and isinstance(y, UnitfulTracer):
        return _ge_tracer(x, y)
    if isinstance(x, UnitfulTracer):
        assert isinstance(y, (StaticArrayLike, Unitful))
        y_unitful = Unitful(val=y) if not isinstance(y, Unitful) else y
        tracer_unit = None if not isinstance(y, Unitful) else y.unit
        y_tracer = UnitfulTracer(unit=tracer_unit, static_unitful=y_unitful, value=y)
        return _ge_tracer(x, y_tracer)
    if isinstance(y, UnitfulTracer):
        assert isinstance(x, (StaticArrayLike, Unitful))
        x_unitful = Unitful(val=x) if not isinstance(x, Unitful) else x
        tracer_unit = None if not isinstance(x, Unitful) else x.unit
        x_tracer = UnitfulTracer(unit=tracer_unit, static_unitful=x_unitful, value=x)
        return _ge_tracer(x_tracer, y)

    # any other array-like
    assert isinstance(x, get_args(AnyArrayLike)), f"Invalid input type for greater than or equal to: {x}"
    assert isinstance(y, get_args(AnyArrayLike)), f"Invalid input type for greater than or equal to: {y}"
    result = x >= y
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

    if isinstance(x, Unitful) and isinstance(y, Unitful):
        return _array_equal_unitful(x, y)
    elif isinstance(x, Unitful) and not isinstance(y, (Unitful, UnitfulTracer)):
        y_unitful = Unitful(val=y)
        return _array_equal_unitful(x, y_unitful)
    elif not isinstance(x, (Unitful, UnitfulTracer)) and isinstance(y, Unitful):
        x_unitful = Unitful(val=x)
        return _array_equal_unitful(x_unitful, y)

    # Tracer handling
    def _array_equal_tracer(x: UnitfulTracer, y: UnitfulTracer):
        assert x.unit == y.unit, f"Unit mismatch for array_equal: {x.unit} vs {y.unit}"
        new_static_unitful = None
        
        if x.static_unitful is not None and y.static_unitful is not None:
            # Recursively call array_equal to compute the static value
            new_static_unitful = array_equal(x.static_unitful, y.static_unitful)
        
        # Output tracer is a dimensionless scalar
        result = UnitfulTracer(unit=None, static_unitful=new_static_unitful)
        node = OperatorNode(
            op_name="array_equal",
            op_kwargs={"x": x, "y": y},
            output=result,
        )
        register_node_full(node)
        return result

    if isinstance(x, UnitfulTracer) and isinstance(y, UnitfulTracer):
        return _array_equal_tracer(x, y)
    if isinstance(x, UnitfulTracer):
        assert isinstance(y, (StaticArrayLike, Unitful))
        y_unitful = Unitful(val=y) if not isinstance(y, Unitful) else y
        tracer_unit = None if not isinstance(y, Unitful) else y.unit
        y_tracer = UnitfulTracer(unit=tracer_unit, static_unitful=y_unitful, value=y)
        return _array_equal_tracer(x, y_tracer)
    if isinstance(y, UnitfulTracer):
        assert isinstance(x, (StaticArrayLike, Unitful))
        x_unitful = Unitful(val=x) if not isinstance(x, Unitful) else x
        tracer_unit = None if not isinstance(x, Unitful) else x.unit
        x_tracer = UnitfulTracer(unit=tracer_unit, static_unitful=x_unitful, value=x)
        return _array_equal_tracer(x_tracer, y)

    # Any other array-like
    assert isinstance(x, get_args(AnyArrayLike)), f"Invalid input type for array_equal: {x}"
    assert isinstance(y, get_args(AnyArrayLike)), f"Invalid input type for array_equal: {y}"
    if isinstance(x, jax.Array) or isinstance(y, jax.Array):
        orig_fn = get_array_equal_original()
    else:
        orig_fn = np.array_equal
    return orig_fn(x, y)
