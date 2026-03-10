from __future__ import annotations

from typing import get_args, overload

import jax
import numpy as np

from quantax.core.typing import AnyArrayLike
from quantax.core.utils import handle_different_scales
from quantax.unitful.tracer import UnitfulTracer
from quantax.unitful.unitful import Unitful

BoolResult = "bool | jax.Array | np.ndarray"


## Equal ###########################
def get_eq_original():
    if hasattr(jax.numpy, "_orig_equal"):
        return jax.numpy._orig_equal
    return jax.numpy.equal


@overload
def eq(x: Unitful, y: Unitful) -> bool | jax.Array | np.ndarray: ...


@overload
def eq(x: AnyArrayLike, y: Unitful) -> bool | jax.Array | np.ndarray: ...


@overload
def eq(x: Unitful, y: AnyArrayLike) -> bool | jax.Array | np.ndarray: ...


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


def eq(x, y):
    def _eq_unitful(x: Unitful, y: Unitful):
        if x.unit != y.unit:
            raise ValueError(f"Cannot compare Unitful arrays with different units: {x.unit} vs {y.unit}")
        _, fx, fy = handle_different_scales(x.scale, y.scale)
        return get_eq_original()(x.val * fx, y.val * fy)

    if isinstance(x, Unitful) and isinstance(y, Unitful):
        return _eq_unitful(x, y)
    elif isinstance(x, Unitful) and not isinstance(y, (Unitful, UnitfulTracer)):
        return _eq_unitful(x, Unitful(val=y))
    elif not isinstance(x, (Unitful, UnitfulTracer)) and isinstance(y, Unitful):
        return _eq_unitful(Unitful(val=x), y)

    if isinstance(x, UnitfulTracer) or isinstance(y, UnitfulTracer):
        x_cmp = x.static_unitful if isinstance(x, UnitfulTracer) and x.static_unitful is not None else x
        y_cmp = y.static_unitful if isinstance(y, UnitfulTracer) and y.static_unitful is not None else y
        if isinstance(x_cmp, Unitful) and isinstance(y_cmp, Unitful):
            return _eq_unitful(x_cmp, y_cmp)
        x_val = x_cmp.val if isinstance(x_cmp, Unitful) else x_cmp
        y_val = y_cmp.val if isinstance(y_cmp, Unitful) else y_cmp
        return get_eq_original()(x_val, y_val)

    assert isinstance(x, get_args(AnyArrayLike)), f"Invalid input type for eq: {x}"
    assert isinstance(y, get_args(AnyArrayLike)), f"Invalid input type for eq: {y}"
    return get_eq_original()(x, y)


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


def ne(x, y):
    def _ne_unitful(x: Unitful, y: Unitful):
        if x.unit != y.unit:
            raise ValueError(f"Cannot compare Unitful arrays with different units: {x.unit} vs {y.unit}")
        _, fx, fy = handle_different_scales(x.scale, y.scale)
        return get_ne_original()(x.val * fx, y.val * fy)

    if isinstance(x, Unitful) and isinstance(y, Unitful):
        return _ne_unitful(x, y)
    elif isinstance(x, Unitful) and not isinstance(y, (Unitful, UnitfulTracer)):
        return _ne_unitful(x, Unitful(val=y))
    elif not isinstance(x, (Unitful, UnitfulTracer)) and isinstance(y, Unitful):
        return _ne_unitful(Unitful(val=x), y)

    if isinstance(x, UnitfulTracer) or isinstance(y, UnitfulTracer):
        x_cmp = x.static_unitful if isinstance(x, UnitfulTracer) and x.static_unitful is not None else x
        y_cmp = y.static_unitful if isinstance(y, UnitfulTracer) and y.static_unitful is not None else y
        if isinstance(x_cmp, Unitful) and isinstance(y_cmp, Unitful):
            return _ne_unitful(x_cmp, y_cmp)
        x_val = x_cmp.val if isinstance(x_cmp, Unitful) else x_cmp
        y_val = y_cmp.val if isinstance(y_cmp, Unitful) else y_cmp
        return get_ne_original()(x_val, y_val)

    assert isinstance(x, get_args(AnyArrayLike)), f"Invalid input type for ne: {x}"
    assert isinstance(y, get_args(AnyArrayLike)), f"Invalid input type for ne: {y}"
    return get_ne_original()(x, y)


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


def lt(x, y):
    def _lt_unitful(x: Unitful, y: Unitful):
        if x.unit != y.unit:
            raise ValueError(f"Cannot compare Unitful arrays with different units: {x.unit} vs {y.unit}")
        _, fx, fy = handle_different_scales(x.scale, y.scale)
        return get_lt_original()(x.val * fx, y.val * fy)

    if isinstance(x, Unitful) and isinstance(y, Unitful):
        return _lt_unitful(x, y)
    elif isinstance(x, Unitful) and not isinstance(y, (Unitful, UnitfulTracer)):
        return _lt_unitful(x, Unitful(val=y))
    elif not isinstance(x, (Unitful, UnitfulTracer)) and isinstance(y, Unitful):
        return _lt_unitful(Unitful(val=x), y)

    if isinstance(x, UnitfulTracer) or isinstance(y, UnitfulTracer):
        x_cmp = x.static_unitful if isinstance(x, UnitfulTracer) and x.static_unitful is not None else x
        y_cmp = y.static_unitful if isinstance(y, UnitfulTracer) and y.static_unitful is not None else y
        if isinstance(x_cmp, Unitful) and isinstance(y_cmp, Unitful):
            return _lt_unitful(x_cmp, y_cmp)
        x_val = x_cmp.val if isinstance(x_cmp, Unitful) else x_cmp
        y_val = y_cmp.val if isinstance(y_cmp, Unitful) else y_cmp
        return get_lt_original()(x_val, y_val)

    assert isinstance(x, get_args(AnyArrayLike)), f"Invalid input type for lt: {x}"
    assert isinstance(y, get_args(AnyArrayLike)), f"Invalid input type for lt: {y}"
    return get_lt_original()(x, y)


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


def le(x, y):
    def _le_unitful(x: Unitful, y: Unitful):
        if x.unit != y.unit:
            raise ValueError(f"Cannot compare Unitful arrays with different units: {x.unit} vs {y.unit}")
        _, fx, fy = handle_different_scales(x.scale, y.scale)
        return get_le_original()(x.val * fx, y.val * fy)

    if isinstance(x, Unitful) and isinstance(y, Unitful):
        return _le_unitful(x, y)
    elif isinstance(x, Unitful) and not isinstance(y, (Unitful, UnitfulTracer)):
        return _le_unitful(x, Unitful(val=y))
    elif not isinstance(x, (Unitful, UnitfulTracer)) and isinstance(y, Unitful):
        return _le_unitful(Unitful(val=x), y)

    if isinstance(x, UnitfulTracer) or isinstance(y, UnitfulTracer):
        x_cmp = x.static_unitful if isinstance(x, UnitfulTracer) and x.static_unitful is not None else x
        y_cmp = y.static_unitful if isinstance(y, UnitfulTracer) and y.static_unitful is not None else y
        if isinstance(x_cmp, Unitful) and isinstance(y_cmp, Unitful):
            return _le_unitful(x_cmp, y_cmp)
        x_val = x_cmp.val if isinstance(x_cmp, Unitful) else x_cmp
        y_val = y_cmp.val if isinstance(y_cmp, Unitful) else y_cmp
        return get_le_original()(x_val, y_val)

    assert isinstance(x, get_args(AnyArrayLike)), f"Invalid input type for le: {x}"
    assert isinstance(y, get_args(AnyArrayLike)), f"Invalid input type for le: {y}"
    return get_le_original()(x, y)


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
def gt(x: int | float | complex | np.number, y: np.ndarray) -> np.ndarray: ...


@overload
def gt(x: np.ndarray, y: int | float | complex | np.number) -> np.ndarray: ...


@overload
def gt(x: int | float | complex | np.number, y: jax.Array) -> jax.Array: ...


@overload
def gt(x: jax.Array, y: int | float | complex | np.number) -> jax.Array: ...


@overload
def gt(x: jax.Array, y: jax.Array) -> jax.Array: ...


@overload
def gt(x: jax.Array, y: np.ndarray) -> jax.Array: ...


@overload
def gt(x: np.ndarray, y: jax.Array) -> jax.Array: ...


@overload
def gt(x: np.ndarray, y: np.ndarray) -> np.ndarray: ...


def gt(x, y):
    def _gt_unitful(x: Unitful, y: Unitful):
        if x.unit != y.unit:
            raise ValueError(f"Cannot compare Unitful arrays with different units: {x.unit} vs {y.unit}")
        _, fx, fy = handle_different_scales(x.scale, y.scale)
        return get_gt_original()(x.val * fx, y.val * fy)

    if isinstance(x, Unitful) and isinstance(y, Unitful):
        return _gt_unitful(x, y)
    elif isinstance(x, Unitful) and not isinstance(y, (Unitful, UnitfulTracer)):
        return _gt_unitful(x, Unitful(val=y))
    elif not isinstance(x, (Unitful, UnitfulTracer)) and isinstance(y, Unitful):
        return _gt_unitful(Unitful(val=x), y)

    if isinstance(x, UnitfulTracer) or isinstance(y, UnitfulTracer):
        x_cmp = x.static_unitful if isinstance(x, UnitfulTracer) and x.static_unitful is not None else x
        y_cmp = y.static_unitful if isinstance(y, UnitfulTracer) and y.static_unitful is not None else y
        if isinstance(x_cmp, Unitful) and isinstance(y_cmp, Unitful):
            return _gt_unitful(x_cmp, y_cmp)
        x_val = x_cmp.val if isinstance(x_cmp, Unitful) else x_cmp
        y_val = y_cmp.val if isinstance(y_cmp, Unitful) else y_cmp
        return get_gt_original()(x_val, y_val)

    assert isinstance(x, get_args(AnyArrayLike)), f"Invalid input type for gt: {x}"
    assert isinstance(y, get_args(AnyArrayLike)), f"Invalid input type for gt: {y}"
    return get_gt_original()(x, y)


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


def ge(x, y):
    def _ge_unitful(x: Unitful, y: Unitful):
        if x.unit != y.unit:
            raise ValueError(f"Cannot compare Unitful arrays with different units: {x.unit} vs {y.unit}")
        _, fx, fy = handle_different_scales(x.scale, y.scale)
        return get_ge_original()(x.val * fx, y.val * fy)

    if isinstance(x, Unitful) and isinstance(y, Unitful):
        return _ge_unitful(x, y)
    elif isinstance(x, Unitful) and not isinstance(y, (Unitful, UnitfulTracer)):
        return _ge_unitful(x, Unitful(val=y))
    elif not isinstance(x, (Unitful, UnitfulTracer)) and isinstance(y, Unitful):
        return _ge_unitful(Unitful(val=x), y)

    if isinstance(x, UnitfulTracer) or isinstance(y, UnitfulTracer):
        x_cmp = x.static_unitful if isinstance(x, UnitfulTracer) and x.static_unitful is not None else x
        y_cmp = y.static_unitful if isinstance(y, UnitfulTracer) and y.static_unitful is not None else y
        if isinstance(x_cmp, Unitful) and isinstance(y_cmp, Unitful):
            return _ge_unitful(x_cmp, y_cmp)
        x_val = x_cmp.val if isinstance(x_cmp, Unitful) else x_cmp
        y_val = y_cmp.val if isinstance(y_cmp, Unitful) else y_cmp
        return get_ge_original()(x_val, y_val)

    assert isinstance(x, get_args(AnyArrayLike)), f"Invalid input type for ge: {x}"
    assert isinstance(y, get_args(AnyArrayLike)), f"Invalid input type for ge: {y}"
    return get_ge_original()(x, y)
