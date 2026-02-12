# ruff: noqa: F811
import cmath
import math
from typing import Sequence, overload

import jax
import jax.numpy as jnp
import numpy as np

from quantax.core.constants import ATOL_COMPARSION, MAX_STATIC_OPTIMIZED_SIZE, RTOL_COMPARSION
from quantax.core.fraction import IntFraction
from quantax.core.glob import register_node
from quantax.core.typing import PHYSICAL_DTYPES, SI, AnyArrayLike
from quantax.core.utils import (
    can_perform_static_ops,
    dim_after_multiplication,
    handle_n_scales,
    is_struct_optimizable,
    is_traced,
    output_unitful_for_array,
)
from quantax.functional.utils import get_static_operand, AnyUnitType
from quantax.unitful.tracer import OperatorNode, UnitfulTracer
from quantax.core.unit import (
    EMPTY_UNIT,
    Unit,
)
from quantax.unitful.unitful import Unitful, can_optimize_scale
from ortools.math_opt.python import mathopt


## Multiplication ###########################
def constraints_multiply(
    x: mathopt.Variable,
    y: mathopt.Variable,
    out: Sequence[mathopt.Variable],
) -> list[mathopt.BoundedLinearExpression]:
    assert len(out) == 1
    return [x + y == out[0]]


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
        node = OperatorNode(
            op_name="multiply",
            args={"x": x, "y": y},
        )
        result = UnitfulTracer(unit=new_unit, parent=node, static_unitful=new_static_unitful)
        node.output_tracer = (result,)
        register_node(node)
        return result
    
    if isinstance(x, UnitfulTracer) and isinstance(y, UnitfulTracer):
        return _mul_tracer(x, y)
    # elif isinstance(x, UnitfulTracer) and not isinstance(y, UnitfulTracer):
    #     y_unitful = y if isinstance(y, Unitful) else Unitful(val=y)
    #     return _mul_tracer(x, y_unitful)
    # elif not isinstance(x, UnitfulTracer) and isinstance(y, UnitfulTracer):
    #     x_unitful = Unitful(val=x)
    #     return _mul_tracer(x_unitful, y)
    
    # any other array-like
    assert isinstance(x, AnyArrayLike)
    assert isinstance(y, AnyArrayLike)
    return x * y


# ## Division ###########################
# @overload
# def divide(x1: Unitful, x2: Unitful) -> Unitful:
#     unit_dict = {k: v for k, v in x1.unit.dim.items()}
#     for k, v in x2.unit.dim.items():
#         if k in unit_dict:
#             unit_dict[k] -= v
#             if unit_dict[k] == 0:
#                 del unit_dict[k]
#         else:
#             unit_dict[k] = -v
#     new_val = x1.val / x2.val
#     new_scale = x1.unit.scale - x2.unit.scale
#     # if static arrays exist, perform div with static arrs
#     new_static_arr = None
#     if is_traced(new_val):
#         x_arr = get_static_operand(x1)
#         y_arr = get_static_operand(x2)
#         if x_arr is not None and y_arr is not None:
#             new_static_arr = x_arr / y_arr
#     return Unitful(val=new_val, unit=Unit(scale=new_scale, dim=unit_dict), static_arr=new_static_arr)


# @overload
# def divide(x1: AnyArrayLike, x2: Unitful) -> Unitful:
#     new_dim = {k: -v for k, v in x2.unit.dim.items()}
#     scale_offset = 0
#     if not is_traced(x1):
#         unit_x1 = Unitful(val=x1, unit=Unit(scale=0, dim={}))
#         x1 = unit_x1.val
#         scale_offset = unit_x1.unit.scale
#     new_val = x1 / x2.val
#     new_scale = scale_offset - x2.unit.scale
#     # if static arrays exist, perform div with static arrs
#     new_static_arr = None
#     if is_traced(new_val):
#         x_arr = get_static_operand(x1)
#         y_arr = get_static_operand(x2)
#         if x_arr is not None and y_arr is not None:
#             new_static_arr = x_arr / y_arr
#     return Unitful(val=new_val, unit=Unit(dim=new_dim, scale=new_scale), static_arr=new_static_arr)


# @overload
# def divide(x1: Unitful, x2: AnyArrayLike) -> Unitful:
#     scale_offset = 0
#     if not is_traced(x2):
#         unit_x2 = Unitful(val=x2, unit=Unit(scale=0, dim={}))
#         x2 = unit_x2.val
#         scale_offset = -unit_x2.unit.scale
#     new_val = x1.val / x2
#     new_scale = x1.unit.scale - scale_offset
#     # if static arrays exist, perform div with static arrs
#     new_static_arr = None
#     if is_traced(new_val):
#         x_arr = get_static_operand(x1)
#         y_arr = get_static_operand(x2)
#         if x_arr is not None and y_arr is not None:
#             new_static_arr = x_arr / y_arr
#     return Unitful(val=new_val, unit=Unit(scale=new_scale, dim=x1.unit.dim), static_arr=new_static_arr)


# @overload
# def divide(x1: int, x2: int) -> float:
#     return x1 / x2


# @overload
# def divide(x1: int, x2: float) -> float:
#     return x1 / x2


# @overload
# def divide(x1: float, x2: int) -> float:
#     return x1 / x2


# @overload
# def divide(x1: float, x2: float) -> float:
#     return x1 / x2


# @overload
# def divide(x1: int | float, x2: complex) -> complex:
#     return x1 / x2


# @overload
# def divide(x1: complex, x2: int | float | complex) -> complex:
#     return x1 / x2


# @overload
# def divide(x: int | float | complex | np.number, y: np.ndarray) -> np.ndarray:
#     return x / y


# @overload
# def divide(x: np.ndarray, y: int | float | complex | np.number) -> np.ndarray:
#     return x / y


# @overload
# def divide(x: int | float | complex | np.number, y: jax.Array) -> jax.Array:
#     return jnp.asarray(x) / y


# @overload
# def divide(x: jax.Array, y: int | float | complex | np.number) -> jax.Array:
#     return x / jnp.asarray(y)


# @overload
# def divide(x1: jax.Array, x2: jax.Array) -> jax.Array:
#     return x1 / x2


# @overload
# def divide(x1: jax.Array, x2: np.ndarray) -> jax.Array:
#     return x1 / x2


# @overload
# def divide(x1: np.ndarray, x2: jax.Array) -> jax.Array:
#     x1_jax = jnp.asarray(x1)
#     return x1_jax / x2


# @overload
# def divide(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
#     return x1 / x2


# @dispatch
# def divide(x1, x2):
#     del x1, x2
#     raise NotImplementedError()


# ## Addition ###########################
# @overload
# def add(x: Unitful, y: Unitful) -> Unitful:
#     if x.unit.dim != y.unit.dim:
#         raise ValueError(f"Cannot add two arrays with units {x.unit} and {y.unit}.")
#     x_align, y_align = align_scales(x, y)
#     new_val = x_align.val + y_align.val
#     # if static arrays exist, perform add with static arrs
#     new_static_arr = None
#     if is_traced(new_val):
#         x_arr = get_static_operand(x_align)
#         y_arr = get_static_operand(y_align)
#         if x_arr is not None and y_arr is not None:
#             new_static_arr = x_arr + y_arr
#     return Unitful(val=new_val, unit=x_align.unit, static_arr=new_static_arr)


# @overload
# def add(x: Unitful, y: AnyArrayLike) -> Unitful:
#     if x.unit.dim:
#         raise ValueError(f"Cannot add non-unitful to array with unit {x.unit}.")
#     y_unitful = Unitful(val=y, unit=Unit(scale=0, dim={}))
#     return add(x, y_unitful)


# @overload
# def add(x: AnyArrayLike, y: Unitful) -> Unitful:
#     return add(y, x)


# @overload
# def add(x: int, y: int) -> int:
#     return x + y


# @overload
# def add(x: int, y: float) -> float:
#     return x + y


# @overload
# def add(x: float, y: int) -> float:
#     return x + y


# @overload
# def add(x: float, y: float) -> float:
#     return x + y


# @overload
# def add(x: int | float, y: complex) -> complex:
#     return x + y


# @overload
# def add(x: complex, y: int | float | complex) -> complex:
#     return x + y


# @overload
# def add(x: int | float | complex | np.number, y: np.ndarray) -> np.ndarray:
#     return x + y


# @overload
# def add(x: np.ndarray, y: int | float | complex | np.number) -> np.ndarray:
#     return x + y


# @overload
# def add(x: int | float | complex | np.number, y: jax.Array) -> jax.Array:
#     return jnp.asarray(x) + y


# @overload
# def add(x: jax.Array, y: int | float | complex | np.number) -> jax.Array:
#     return x + jnp.asarray(y)


# @overload
# def add(x: jax.Array, y: jax.Array) -> jax.Array:
#     return x + y


# @overload
# def add(x: jax.Array, y: np.ndarray) -> jax.Array:
#     return x + y


# @overload
# def add(x: np.ndarray, y: jax.Array) -> jax.Array:
#     x_jax = jnp.asarray(x)
#     return x_jax + y


# @overload
# def add(x: np.ndarray, y: np.ndarray) -> np.ndarray:
#     return x + y


# @dispatch
# def add(x, y):
#     del x, y
#     raise NotImplementedError()


# ## Matrix Multiplication ###################################
# @overload
# def matmul(x: Unitful, y: Unitful, **kwargs) -> Unitful:
#     x_align, y_align = align_scales(x, y)

#     invalid_types = (bool, int, float, complex, np.bool_, np.number)
#     if isinstance(x.val, invalid_types) or isinstance(y.val, invalid_types):
#         raise TypeError(f"matmul received invalid types: {type(x).__name__}, {type(y).__name__}")
#     # handling of numpy arrays
#     elif isinstance(x.val, np.ndarray) and isinstance(y.val, np.ndarray):
#         new_val = np.matmul(x_align.val, y_align.val, **kwargs)
#     else:
#         new_val = jnp._orig_matmul(x_align.val, y_align.val, **kwargs)  # type: ignore

#     unit_dict = dim_after_multiplication(x.unit.dim, y.unit.dim)
#     new_scale = 2 * x_align.unit.scale
#     # if static arrays exist, perform subtract with static arrs
#     new_static_arr = None
#     if is_traced(new_val):
#         x_arr = get_static_operand(x_align)
#         y_arr = get_static_operand(y_align)
#         if can_perform_static_ops(x_arr) and can_perform_static_ops(y_arr):
#             new_static_arr = np.matmul(x_arr, y_arr, **kwargs)  # type: ignore
#     return Unitful(val=new_val, unit=Unit(scale=new_scale, dim=unit_dict), static_arr=new_static_arr)


# @overload
# def matmul(x: jax.Array, y: jax.Array, **kwargs) -> jax.Array:
#     return jnp._orig_matmul(x, y, **kwargs)  # type: ignore


# @overload
# def matmul(x: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
#     return np.matmul(x, y, **kwargs)


# @overload
# def matmul(x: jax.Array, y: np.ndarray, **kwargs) -> np.ndarray:
#     return jnp._orig_matmul(x, y, **kwargs)  # type: ignore


# @overload
# def matmul(x: np.ndarray, y: jax.Array, **kwargs) -> np.ndarray:
#     return jnp._orig_matmul(x, y, **kwargs)  # type: ignore


# @dispatch
# def matmul(x, y):
#     del x, y
#     raise NotImplementedError()


# ## Subtractions ###########################
# @overload
# def subtract(x: Unitful, y: Unitful) -> Unitful:
#     if x.unit.dim != y.unit.dim:
#         raise ValueError(f"Cannot subtract two arrays with units {x.unit} and {y.unit}.")
#     x_align, y_align = align_scales(x, y)
#     if isinstance(x_align.val, np.bool) or isinstance(y_align.val, np.bool):
#         raise Exception(f"Subtract not supported for bool: {x}, {y}")
#     new_val = x_align.val - y_align.val
#     # if static arrays exist, perform subtract with static arrs
#     new_static_arr = None
#     if is_traced(new_val):
#         x_arr = get_static_operand(x_align)
#         y_arr = get_static_operand(y_align)
#         if can_perform_static_ops(x_arr) and can_perform_static_ops(y_arr):
#             new_static_arr = x_arr - y_arr  # type: ignore
#     return Unitful(val=new_val, unit=x_align.unit, static_arr=new_static_arr)


# @overload
# def subtract(x: Unitful, y: AnyArrayLike) -> Unitful:
#     if x.unit.dim:
#         raise ValueError(f"Cannot add non-unitful to array with unit {x.unit}.")
#     y_unitful = Unitful(val=y, unit=EMPTY_UNIT)
#     return subtract(x, y_unitful)


# @overload
# def subtract(x: AnyArrayLike, y: Unitful) -> Unitful:
#     if y.unit.dim:
#         raise ValueError(f"Cannot add non-unitful to array with unit {y.unit}.")
#     x_unitful = Unitful(val=x, unit=EMPTY_UNIT)
#     return subtract(x_unitful, y)


# @overload
# def subtract(x: int, y: int) -> int:
#     return x - y


# @overload
# def subtract(x: int, y: float) -> float:
#     return x - y


# @overload
# def subtract(x: float, y: int) -> float:
#     return x - y


# @overload
# def subtract(x: float, y: float) -> float:
#     return x - y


# @overload
# def subtract(x: int | float, y: complex) -> complex:
#     return x - y


# @overload
# def subtract(x: complex, y: int | float | complex) -> complex:
#     return x - y


# @overload
# def subtract(x: int | float | complex | np.number, y: np.ndarray) -> np.ndarray:
#     return x - y


# @overload
# def subtract(x: np.ndarray, y: int | float | complex | np.number) -> np.ndarray:
#     return x - y


# @overload
# def subtract(x: int | float | complex | np.number, y: jax.Array) -> jax.Array:
#     return jnp.asarray(x) - y


# @overload
# def subtract(x: jax.Array, y: int | float | complex | np.number) -> jax.Array:
#     return x - jnp.asarray(y)


# @overload
# def subtract(x: jax.Array, y: jax.Array) -> jax.Array:
#     return x - y


# @overload
# def subtract(x: jax.Array, y: np.ndarray) -> jax.Array:
#     return x - y


# @overload
# def subtract(x: np.ndarray, y: jax.Array) -> jax.Array:
#     x_jax = jnp.asarray(x)
#     return x_jax - y


# @overload
# def subtract(x: np.ndarray, y: np.ndarray) -> np.ndarray:
#     return x - y


# @dispatch
# def subtract(x, y):
#     del x, y
#     raise NotImplementedError()


# ## less than ##########################
# @overload
# def lt(
#     x: Unitful,
#     y: Unitful,
# ) -> Unitful:
#     x_align, y_align = align_scales(x, y)
#     if isinstance(x_align.val, complex) or isinstance(y_align.val, complex):
#         raise Exception(f"Cannot compare complex values: {x}, {y}")
#     new_val = x_align.val < y_align.val
#     new_static_arr = None
#     if is_traced(new_val):
#         x_arr = get_static_operand(x_align)
#         y_arr = get_static_operand(y_align)
#         if x_arr is not None and y_arr is not None:
#             assert not isinstance(x_arr, complex) and not isinstance(y_arr, complex)
#             new_static_arr = x_arr < y_arr
#     if output_unitful_for_array(new_static_arr):
#         return Unitful(val=new_val, unit=EMPTY_UNIT, static_arr=new_static_arr)
#     return Unitful(val=new_val, unit=EMPTY_UNIT)


# @overload
# def lt(
#     x: Unitful,
#     y: AnyArrayLike,
# ) -> Unitful:
#     assert x.unit.dim == {}, f"Cannot compare unitful with dim {x.unit.dim} to unitless quantity"
#     result = lt(x, Unitful(val=y, unit=EMPTY_UNIT))
#     return result


# @overload
# def lt(
#     x: AnyArrayLike,
#     y: Unitful,
# ) -> Unitful:
#     return ge(y, x)


# @overload
# def lt(x: int | float, y: int | float) -> bool:
#     return x < y


# @overload
# def lt(x: jax.Array, y: jax.Array) -> jax.Array:
#     return x < y


# @overload
# def lt(x: np.ndarray, y: np.ndarray) -> np.ndarray:
#     return x < y


# @overload
# def lt(x: np.number, y: np.number) -> np.bool:
#     return x < y


# @dispatch
# def lt(x, y):
#     del x, y
#     raise NotImplementedError()


# ## less equal ##########################
# @overload
# def le(
#     x: Unitful,
#     y: Unitful,
# ) -> Unitful:
#     x_align, y_align = align_scales(x, y)
#     if isinstance(x_align.val, complex) or isinstance(y_align.val, complex):
#         raise Exception(f"Cannot compare complex values: {x}, {y}")

#     if isinstance(x_align.val, jax.Array) or isinstance(y_align.val, jax.Array):
#         new_val = jnp.logical_or(
#             jnp.isclose(x_align.val, y_align.val, rtol=RTOL_COMPARSION, atol=ATOL_COMPARSION), x_align.val < y_align.val
#         )
#     elif isinstance(x_align.val, np.ndarray | np.number) or isinstance(y_align.val, np.ndarray | np.number):
#         new_val = np.logical_or(
#             np.isclose(x_align.val, y_align.val, rtol=RTOL_COMPARSION, atol=ATOL_COMPARSION), x_align.val < y_align.val
#         )
#     else:
#         new_val = x_align.val <= y_align.val

#     new_static_arr = None
#     if is_traced(new_val):
#         x_arr = get_static_operand(x_align)
#         y_arr = get_static_operand(y_align)
#         if x_arr is not None and y_arr is not None:
#             assert not isinstance(x_arr, complex) and not isinstance(y_arr, complex)
#             new_static_arr = np.logical_or(
#                 np.isclose(x_arr, y_arr, rtol=RTOL_COMPARSION, atol=ATOL_COMPARSION), x_arr < y_arr
#             )
#     if output_unitful_for_array(new_static_arr):
#         return Unitful(val=new_val, unit=EMPTY_UNIT, static_arr=new_static_arr)
#     return Unitful(val=new_val, unit=EMPTY_UNIT)


# @overload
# def le(
#     x: Unitful,
#     y: AnyArrayLike,
# ) -> Unitful:
#     assert x.unit.dim == {}, f"Cannot compare unitful with dim {x.unit.dim} to unitless quantity"
#     result = le(x, Unitful(val=y, unit=EMPTY_UNIT))
#     return result


# @overload
# def le(
#     x: AnyArrayLike,
#     y: Unitful,
# ) -> Unitful:
#     return gt(y, x)


# @overload
# def le(x: int | float, y: int | float) -> bool:
#     return x <= y


# @overload
# def le(x: jax.Array, y: jax.Array) -> jax.Array:
#     return x <= y


# @overload
# def le(x: np.ndarray, y: np.ndarray) -> np.ndarray:
#     return x <= y


# @overload
# def le(x: np.number, y: np.number) -> np.bool:
#     return x <= y


# @dispatch
# def le(x, y):
#     del x, y
#     raise NotImplementedError()


# ## equal ##########################
# @overload
# def eq(
#     x: Unitful,
#     y: Unitful,
# ) -> Unitful:
#     x_align, y_align = align_scales(x, y)

#     if isinstance(x_align.val, jax.Array) or isinstance(y_align.val, jax.Array):
#         new_val = jnp.isclose(x_align.val, y_align.val, rtol=RTOL_COMPARSION, atol=ATOL_COMPARSION)
#     elif isinstance(x_align.val, np.ndarray | np.number) or isinstance(y_align.val, np.ndarray | np.number):
#         new_val = np.isclose(x_align.val, y_align.val, rtol=RTOL_COMPARSION, atol=ATOL_COMPARSION)
#     else:
#         new_val = x_align.val == y_align.val

#     new_static_arr = None
#     if is_traced(new_val):
#         x_arr = get_static_operand(x_align)
#         y_arr = get_static_operand(y_align)
#         if x_arr is not None and y_arr is not None:
#             new_static_arr = np.isclose(x_arr, y_arr, rtol=RTOL_COMPARSION, atol=ATOL_COMPARSION)
#     if output_unitful_for_array(new_static_arr):
#         return Unitful(val=new_val, unit=EMPTY_UNIT, static_arr=new_static_arr)
#     return Unitful(val=new_val, unit=EMPTY_UNIT)


# @overload
# def eq(
#     x: Unitful,
#     y: AnyArrayLike,
# ) -> Unitful:
#     assert x.unit.dim == {}, f"Cannot compare unitful with dim {x.unit.dim} to unitless quantity"
#     result = eq(x, Unitful(val=y, unit=EMPTY_UNIT))
#     return result


# @overload
# def eq(
#     x: AnyArrayLike,
#     y: Unitful,
# ) -> Unitful:
#     return eq(y, x)


# @overload
# def eq(x: int | float, y: int | float) -> bool:
#     return x == y


# @overload
# def eq(x: jax.Array, y: jax.Array) -> jax.Array:
#     return x == y


# @overload
# def eq(x: np.ndarray, y: np.ndarray) -> np.ndarray:
#     return x == y


# @overload
# def eq(x: np.number, y: np.number) -> np.bool:
#     return x == y


# @dispatch
# def eq(x, y):
#     del x, y
#     raise NotImplementedError()


# ## not equal ##########################
# @overload
# def ne(
#     x: Unitful,
#     y: Unitful,
# ) -> Unitful:
#     x_align, y_align = align_scales(x, y)
#     new_val = x_align.val != y_align.val
#     new_static_arr = None
#     if is_traced(new_val):
#         x_arr = get_static_operand(x_align)
#         y_arr = get_static_operand(y_align)
#         if x_arr is not None and y_arr is not None:
#             new_static_arr = x_arr != y_arr
#     if output_unitful_for_array(new_static_arr):
#         return Unitful(val=new_val, unit=EMPTY_UNIT, static_arr=new_static_arr)
#     return Unitful(val=new_val, unit=EMPTY_UNIT)


# @overload
# def ne(
#     x: Unitful,
#     y: AnyArrayLike,
# ) -> Unitful:
#     assert x.unit.dim == {}, f"Cannot compare unitful with dim {x.unit.dim} to unitless quantity"
#     result = ne(x, Unitful(val=y, unit=EMPTY_UNIT))
#     return result


# @overload
# def ne(
#     x: AnyArrayLike,
#     y: Unitful,
# ) -> Unitful:
#     return ne(y, x)


# @overload
# def ne(x: int | float, y: int | float) -> bool:
#     return x != y


# @overload
# def ne(x: jax.Array, y: jax.Array) -> jax.Array:
#     return x != y


# @overload
# def ne(x: np.ndarray, y: np.ndarray) -> np.ndarray:
#     return x != y


# @overload
# def ne(x: np.number, y: np.number) -> np.bool:
#     return x != y


# @dispatch
# def ne(x, y):
#     del x, y
#     raise NotImplementedError()


# ## greater equal ##########################
# @overload
# def ge(
#     x: Unitful,
#     y: Unitful,
# ) -> Unitful:
#     x_align, y_align = align_scales(x, y)
#     if isinstance(x_align.val, complex) or isinstance(y_align.val, complex):
#         raise Exception(f"Cannot compare complex values: {x}, {y}")

#     if isinstance(x_align.val, jax.Array) or isinstance(y_align.val, jax.Array):
#         new_val = jnp.logical_or(
#             jnp.isclose(x_align.val, y_align.val, rtol=RTOL_COMPARSION, atol=ATOL_COMPARSION), x_align.val > y_align.val
#         )
#     elif isinstance(x_align.val, np.ndarray | np.number) or isinstance(y_align.val, np.ndarray | np.number):
#         new_val = np.logical_or(
#             np.isclose(x_align.val, y_align.val, rtol=RTOL_COMPARSION, atol=ATOL_COMPARSION), x_align.val > y_align.val
#         )
#     else:
#         new_val = x_align.val >= y_align.val

#     new_static_arr = None
#     if is_traced(new_val):
#         x_arr = get_static_operand(x_align)
#         y_arr = get_static_operand(y_align)
#         if x_arr is not None and y_arr is not None:
#             assert not isinstance(x_arr, complex) and not isinstance(y_arr, complex)
#             new_static_arr = np.logical_or(
#                 np.isclose(x_arr, y_arr, rtol=RTOL_COMPARSION, atol=ATOL_COMPARSION), x_arr > y_arr
#             )
#     if output_unitful_for_array(new_static_arr):
#         return Unitful(val=new_val, unit=EMPTY_UNIT, static_arr=new_static_arr)
#     return Unitful(val=new_val, unit=EMPTY_UNIT)


# @overload
# def ge(
#     x: Unitful,
#     y: AnyArrayLike,
# ) -> Unitful:
#     assert x.unit.dim == {}, f"Cannot compare unitful with dim {x.unit.dim} to unitless quantity"
#     result = ge(x, Unitful(val=y, unit=EMPTY_UNIT))
#     return result


# @overload
# def ge(
#     x: AnyArrayLike,
#     y: Unitful,
# ) -> Unitful:
#     return lt(y, x)


# @overload
# def ge(x: int | float, y: int | float) -> bool:
#     return x >= y


# @overload
# def ge(x: jax.Array, y: jax.Array) -> jax.Array:
#     return x >= y


# @overload
# def ge(x: np.ndarray, y: np.ndarray) -> np.ndarray:
#     return x >= y


# @overload
# def ge(x: np.number, y: np.number) -> np.bool:
#     return x >= y


# @dispatch
# def ge(x, y):
#     del x, y
#     raise NotImplementedError()


# ## greater than ##########################
# @overload
# def gt(
#     x: Unitful,
#     y: Unitful,
# ) -> Unitful:
#     x_align, y_align = align_scales(x, y)
#     if isinstance(x_align.val, complex) or isinstance(y_align.val, complex):
#         raise Exception(f"Cannot compare complex values: {x}, {y}")
#     new_val = x_align.val > y_align.val
#     new_static_arr = None
#     if is_traced(new_val):
#         x_arr = get_static_operand(x_align)
#         y_arr = get_static_operand(y_align)
#         if x_arr is not None and y_arr is not None:
#             assert not isinstance(x_arr, complex) and not isinstance(y_arr, complex)
#             new_static_arr = x_arr > y_arr
#     if output_unitful_for_array(new_static_arr):
#         return Unitful(val=new_val, unit=EMPTY_UNIT, static_arr=new_static_arr)
#     return Unitful(val=new_val, unit=EMPTY_UNIT)


# @overload
# def gt(
#     x: Unitful,
#     y: AnyArrayLike,
# ) -> Unitful:
#     assert x.unit.dim == {}, f"Cannot compare unitful with dim {x.unit.dim} to unitless quantity"
#     result = gt(x, Unitful(val=y, unit=EMPTY_UNIT))
#     return result


# @overload
# def gt(
#     x: AnyArrayLike,
#     y: Unitful,
# ) -> Unitful:
#     return le(y, x)


# @overload
# def gt(x: int | float, y: int | float) -> bool:
#     return x > y


# @overload
# def gt(x: jax.Array, y: jax.Array) -> jax.Array:
#     return x > y


# @overload
# def gt(x: np.ndarray, y: np.ndarray) -> np.ndarray:
#     return x > y


# @overload
# def gt(x: np.number, y: np.number) -> np.bool:
#     return x > y


# @dispatch
# def gt(x, y):
#     del x, y
#     raise NotImplementedError()


# ## power #######################################
# @overload
# def pow(x: Unitful, y: int) -> Unitful:
#     new_dim = {}
#     for k, v in x.unit.dim.items():
#         new_dim[k] = v * y

#     if isinstance(x.val, jax.Array):
#         new_val = jnp.power(x.val, y)
#     elif isinstance(x.val, np.ndarray | np.number | np.bool_):
#         new_val = np.power(x.val, y)
#     else:  # int, float, complex, bool
#         new_val = x.val**y

#     new_scale = x.unit.scale * y
#     new_static_arr = None
#     if is_traced(new_val):
#         x_arr = get_static_operand(x)
#         if x_arr is not None:
#             new_static_arr = x_arr**y
#     return Unitful(val=new_val, unit=Unit(scale=new_scale, dim=new_dim), static_arr=new_static_arr)


# @overload
# def pow(x: int, y: int) -> int:
#     return x**y


# @overload
# def pow(x: int, y: float) -> float:
#     return x**y


# @overload
# def pow(x: float, y: int) -> float:
#     return x**y


# @overload
# def pow(x: float, y: float) -> float:
#     return x**y


# @overload
# def pow(x: complex, y: float | int) -> complex:
#     return x**y


# @overload
# def pow(x: float | int | complex, y: complex) -> complex:
#     return x**y


# @overload
# def pow(x: int | float | complex | np.number, y: np.ndarray) -> np.ndarray:
#     return x**y


# @overload
# def pow(x: np.ndarray, y: int | float | complex | np.number) -> np.ndarray:
#     return x**y


# @overload
# def pow(x: int | float | complex | np.number, y: jax.Array) -> jax.Array:
#     x_jax = jnp.asarray(x)
#     return x_jax**y


# @overload
# def pow(x: jax.Array, y: int | float | complex | np.number) -> jax.Array:
#     return x**y


# @overload
# def pow(x: jax.Array, y: jax.Array) -> jax.Array:
#     return x**y


# @overload
# def pow(x: np.ndarray, y: np.ndarray) -> np.ndarray:
#     return x**y


# @overload
# def pow(x: jax.Array, y: np.ndarray) -> jax.Array:
#     return x**y


# @overload
# def pow(x: np.ndarray, y: jax.Array) -> jax.Array:
#     x_jax = jnp.array(x)
#     return x_jax**y


# @dispatch
# def pow(x, y):
#     del x, y
#     raise NotImplementedError()


# ## unary operation #######################################
# def unary_fn(x: Unitful, op_str: str, *args, **kwargs) -> Unitful:
#     if isinstance(x.val, jax.Array):
#         orig_fn = getattr(jax.numpy, f"_orig_{op_str}")
#         new_val = orig_fn(x.val, *args, **kwargs)

#         if not isinstance(new_val, jax.Array):
#             raise Exception(f"This is an internal error: {op_str} produced {type(new_val)}")
#     else:  # For NumPy types and other non-JAX types, convert them to NumPy first.
#         np_fn = getattr(np, op_str)
#         new_val = np_fn(x.val, *args, **kwargs)

#         if not isinstance(new_val, (np.ndarray, np.number)):
#             raise Exception(f"This is an internal error: {op_str} produced {type(new_val)}")

#     new_static_arr = None
#     if is_traced(new_val):
#         x_arr = get_static_operand(x)
#         if x_arr is not None:
#             np_fn = getattr(np, op_str)
#             new_static_arr = np_fn(x_arr, *args, **kwargs)

#     if new_val.dtype not in PHYSICAL_DTYPES:
#         unit = EMPTY_UNIT
#     else:
#         unit = x.unit

#     return Unitful(val=new_val, unit=unit, static_arr=new_static_arr)


# ## min #######################################
# @overload
# def min(x: Unitful, *args, **kwargs) -> Unitful:
#     return unary_fn(x, "min", *args, **kwargs)


# @overload
# def min(x: jax.Array, *args, **kwargs) -> jax.Array:
#     result_shape_dtype = jax.eval_shape(jnp._orig_min, x, *args, **kwargs)  # type: ignore
#     if output_unitful_for_array(result_shape_dtype):
#         unit_result = unary_fn(Unitful(val=x, unit=EMPTY_UNIT), "min", *args, **kwargs)
#         return unit_result  # type: ignore
#     return jnp._orig_min(x, *args, **kwargs)  # type: ignore


# @overload
# def min(x: np.number, *args, **kwargs) -> np.number:
#     return np.min(x, *args, **kwargs)


# @overload
# def min(x: float | complex | np.ndarray, *args, **kwargs) -> np.ndarray:
#     return np.min(x, *args, **kwargs)


# @dispatch
# def min(x, *args, **kwargs):
#     del x, args, kwargs
#     raise NotImplementedError()


# ## max #######################################
# @overload
# def max(x: Unitful, *args, **kwargs) -> Unitful:
#     return unary_fn(x, "max", *args, **kwargs)


# @overload
# def max(x: jax.Array, *args, **kwargs) -> jax.Array:
#     result_shape_dtype = jax.eval_shape(jnp._orig_max, x, *args, **kwargs)  # type: ignore
#     if output_unitful_for_array(result_shape_dtype):
#         unit_result = unary_fn(Unitful(val=x, unit=EMPTY_UNIT), "max", *args, **kwargs)
#         return unit_result  # type: ignore
#     return jnp._orig_max(x, *args, **kwargs)  # type: ignore


# @overload
# def max(x: np.number, *args, **kwargs) -> np.number:
#     return np.max(x, *args, **kwargs)


# @overload
# def max(x: float | complex | np.ndarray, *args, **kwargs) -> np.ndarray:
#     return np.max(x, *args, **kwargs)


# @dispatch
# def max(x, *args, **kwargs):
#     del x, args, kwargs
#     raise NotImplementedError()


# ## mean #######################################
# @overload
# def mean(x: Unitful, *args, **kwargs) -> Unitful:
#     return unary_fn(x, "mean", *args, **kwargs)


# @overload
# def mean(x: jax.Array, *args, **kwargs) -> jax.Array:
#     result_shape_dtype = jax.eval_shape(jnp._orig_mean, x, *args, **kwargs)  # type: ignore
#     if output_unitful_for_array(result_shape_dtype):
#         unit_result = unary_fn(Unitful(val=x, unit=EMPTY_UNIT), "mean", *args, **kwargs)
#         return unit_result  # type: ignore
#     return jnp._orig_mean(x, *args, **kwargs)  # type: ignore


# @overload
# def mean(x: np.number, *args, **kwargs) -> np.number:
#     return np.mean(x, *args, **kwargs)


# @overload
# def mean(x: float | complex | np.ndarray, *args, **kwargs) -> np.ndarray:
#     return np.mean(x, *args, **kwargs)


# @dispatch
# def mean(x, *args, **kwargs):
#     del x, args, kwargs
#     raise NotImplementedError()


# ## sum #######################################
# @overload
# def sum(x: Unitful, *args, **kwargs) -> Unitful:
#     return unary_fn(x, "sum", *args, **kwargs)


# @overload
# def sum(x: jax.Array, *args, **kwargs) -> jax.Array:
#     result_shape_dtype = jax.eval_shape(jnp._orig_sum, x, *args, **kwargs)  # type: ignore
#     if output_unitful_for_array(result_shape_dtype):
#         unit_result = unary_fn(Unitful(val=x, unit=EMPTY_UNIT), "sum", *args, **kwargs)
#         return unit_result  # type: ignore
#     return jnp._orig_sum(x, *args, **kwargs)  # type: ignore


# @overload
# def sum(x: np.number, *args, **kwargs) -> np.number:
#     return np.sum(x, *args, **kwargs)


# @overload
# def sum(x: float | complex | np.ndarray, *args, **kwargs) -> np.ndarray:
#     return np.sum(x, *args, **kwargs)


# @dispatch
# def sum(x, *args, **kwargs):
#     del x, args, kwargs
#     raise NotImplementedError()


# ## prod #######################################
# @overload
# def prod(x: Unitful, *args, **kwargs) -> Unitful:
#     if x.unit.dim != {}:
#         raise NotImplementedError()
#     # TODO: make numerically more stable by avoiding .value()
#     new_static_arr = x.static_value()
#     new_val = x.value()
#     unary_input = Unitful(val=new_val, unit=EMPTY_UNIT, optimize_scale=False, static_arr=new_static_arr)
#     return unary_fn(unary_input, "prod", *args, **kwargs)


# @overload
# def prod(x: jax.Array, *args, **kwargs) -> jax.Array:
#     result_shape_dtype = jax.eval_shape(jnp._orig_prod, x, *args, **kwargs)  # type: ignore
#     if output_unitful_for_array(result_shape_dtype):
#         unit_result = unary_fn(Unitful(val=x, unit=EMPTY_UNIT), "sum", *args, **kwargs)
#         return unit_result  # type: ignore
#     return jnp._orig_prod(x, *args, **kwargs)  # type: ignore


# @overload
# def prod(x: np.number, *args, **kwargs) -> np.number:
#     return np.prod(x, *args, **kwargs)


# @overload
# def prod(x: float | complex | np.ndarray, *args, **kwargs) -> np.ndarray:
#     return np.prod(x, *args, **kwargs)


# @dispatch
# def prod(x, *args, **kwargs):
#     del x, args, kwargs
#     raise NotImplementedError()


# ## abs #######################################
# @overload
# def abs_impl(x: Unitful) -> Unitful:
#     return unary_fn(x, "abs")


# @overload
# def abs_impl(x: jax.Array) -> jax.Array:
#     return jnp._orig_abs(x)  # type: ignore


# @overload
# def abs_impl(x: np.number) -> np.number:
#     return np.abs(x)


# @overload
# def abs_impl(x: np.ndarray) -> np.ndarray:
#     return np.abs(x)


# @overload
# def abs_impl(x: int) -> int:
#     return abs(x)


# @overload
# def abs_impl(x: float) -> float:
#     return abs(x)


# @overload
# def abs_impl(x: complex) -> complex:
#     return abs(x)


# @dispatch
# def abs_impl(x):
#     del x
#     raise NotImplementedError()


# ## astype #######################################
# @overload
# def astype(x: Unitful, dtype, *args, **kwargs) -> Unitful:
#     if isinstance(x.val, int | float | complex):
#         raise TypeError(f"python scalar of type {type(x.val)} is not a valid input for astype")
#     return unary_fn(x, "astype", dtype, *args, **kwargs)


# @overload
# def astype(x: jax.Array, *args, **kwargs) -> jax.Array:
#     partial_fn = jax.tree_util.Partial(jnp._orig_astype, x, *args, **kwargs)  # type: ignore
#     result_shape_dtype = jax.eval_shape(partial_fn)
#     if output_unitful_for_array(result_shape_dtype):
#         unit_result = unary_fn(Unitful(val=x, unit=EMPTY_UNIT), "astype", *args, **kwargs)
#         return unit_result  # type: ignore
#     return jnp._orig_astype(x, *args, **kwargs)  # type: ignore


# @overload
# def astype(x: np.number, dtype, *args, **kwargs) -> np.ndarray:
#     x_array = np.asarray(x)
#     return x_array.astype(dtype, *args, **kwargs)


# @overload
# def astype(x: np.ndarray, dtype, *args, **kwargs) -> np.ndarray:
#     return x.astype(dtype, *args, **kwargs)


# @dispatch
# def astype(x, *args, **kwargs):
#     del x, args, kwargs
#     raise NotImplementedError()


# ## squeeze #######################################
# @overload
# def squeeze(x: Unitful, *args, **kwargs) -> Unitful:
#     if isinstance(x.val, int | float | complex):
#         raise TypeError(f"python scalar of type {type(x.val)} is not a valid input for squeeze")
#     return unary_fn(x, "squeeze", *args, **kwargs)


# @overload
# def squeeze(x: jax.Array, *args, **kwargs) -> jax.Array:
#     return jnp._orig_squeeze(x, *args, **kwargs)  # type: ignore


# @overload
# def squeeze(x: np.number, *args, **kwargs) -> np.number:
#     return np.squeeze(x, *args, **kwargs)


# @overload
# def squeeze(x: np.ndarray, *args, **kwargs) -> np.ndarray:
#     return np.squeeze(x, *args, **kwargs)


# @dispatch
# def squeeze(x, *args, **kwargs):
#     del x, args, kwargs
#     raise NotImplementedError()


# ## reshape #######################################
# @overload
# def reshape(x: Unitful, shape, *args, **kwargs) -> Unitful:
#     if isinstance(x.val, int | float | complex):
#         raise TypeError(f"python scalar of type {type(x.val)} is not a valid input for reshape")
#     return unary_fn(x, "reshape", shape, *args, **kwargs)


# @overload
# def reshape(x: jax.Array, shape, *args, **kwargs) -> jax.Array:
#     return jnp._orig_reshape(x, shape, *args, **kwargs)  # type: ignore


# @overload
# def reshape(x: np.number | np.ndarray, shape, *args, **kwargs) -> np.ndarray:
#     if isinstance(x, np.number):
#         x = np.asarray(x)
#     return x.reshape(shape, *args, **kwargs)


# @dispatch
# def reshape(x, *args, **kwargs):
#     del x, args, kwargs
#     raise NotImplementedError()


# ## argmax #######################################
# @overload
# def argmax(x: Unitful, *args, **kwargs) -> Unitful:
#     return unary_fn(x, "argmax", *args, **kwargs)


# @overload
# def argmax(x: jax.Array, *args, **kwargs) -> jax.Array:
#     return jnp._orig_argmax(x, *args, **kwargs)  # type: ignore


# @overload
# def argmax(x: np.ndarray, *args, **kwargs) -> np.ndarray:
#     return np.argmax(x, *args, **kwargs)


# @dispatch
# def argmax(x, *args, **kwargs):
#     del x, args, kwargs
#     raise NotImplementedError()


# ## argmin #######################################
# @overload
# def argmin(x: Unitful, *args, **kwargs) -> Unitful:
#     return unary_fn(x, "argmin", *args, **kwargs)


# @overload
# def argmin(x: jax.Array, *args, **kwargs) -> jax.Array:
#     return jnp._orig_argmin(x, *args, **kwargs)  # type: ignore


# @overload
# def argmin(x: np.ndarray, *args, **kwargs) -> np.ndarray:
#     return np.argmin(x, *args, **kwargs)


# @dispatch
# def argmin(x, *args, **kwargs):
#     del x, args, kwargs
#     raise NotImplementedError()


# ## Square Root ###########################
# @overload
# def sqrt(
#     x: Unitful,
# ) -> Unitful:
#     new_dim: dict[SI, int | IntFraction] = {}
#     for k, v in x.unit.dim.items():
#         if isinstance(v, int):
#             if v % 2 == 0:
#                 new_dim[k] = v // 2
#             else:
#                 new_dim[k] = IntFraction(v, 2)
#         elif isinstance(v, IntFraction):
#             new_dim[k] = v / 2
#         else:
#             raise Exception(f"Invalid dimension exponent: {v}")
#     if x.unit.scale % 2 == 0:
#         new_val = jnp._orig_sqrt(x.val)  # type: ignore
#         new_scale = x.unit.scale // 2
#     else:
#         new_val = jnp._orig_sqrt(x.val) * math.sqrt(10)  # type: ignore
#         new_scale = math.floor(x.unit.scale / 2)
#     # static arr computation
#     new_static_arr = None
#     if is_traced(new_val):
#         x_arr = get_static_operand(x)
#         if x_arr is not None:
#             new_static_arr = np.sqrt(x_arr)
#             if x.unit.scale % 2 != 0:
#                 new_static_arr = new_static_arr * math.sqrt(10)
#     return Unitful(val=new_val, unit=Unit(scale=new_scale, dim=new_dim), static_arr=new_static_arr)


# @overload
# def sqrt(x: int | float) -> float:
#     return math.sqrt(x)


# @overload
# def sqrt(x: jax.Array) -> jax.Array:
#     return jnp._orig_sqrt(x)  # type: ignore


# @dispatch
# def sqrt(x):
#     del x
#     raise NotImplementedError()


# ## Roll #####################################
# @overload
# def roll(
#     x: Unitful,
#     *args,
#     **kwargs,
# ) -> Unitful:
#     new_val = jnp._orig_roll(x.val, *args, **kwargs)  # type: ignore
#     # static arr computation
#     new_static_arr = None
#     if is_traced(new_val):
#         x_arr = get_static_operand(x)
#         if x_arr is not None:
#             new_static_arr = np.roll(x_arr, *args, **kwargs)
#     return Unitful(val=new_val, unit=x.unit, static_arr=new_static_arr)


# @overload
# def roll(
#     x: jax.Array,
#     *args,
#     **kwargs,
# ) -> jax.Array:
#     return jnp._orig_roll(x, *args, **kwargs)  # type: ignore


# @dispatch
# def roll(
#     x,
#     *args,
#     **kwargs,
# ):
#     del x, args, kwargs
#     raise NotImplementedError()


# ## Square #####################################
# @overload
# def square(
#     x: Unitful,
# ) -> Unitful:
#     return x * x


# @overload
# def square(x: int | float) -> float:
#     return x * x


# @overload
# def square(x: complex) -> complex:
#     return x * x


# @overload
# def square(x: jax.Array) -> jax.Array:
#     return x * x


# @dispatch
# def square(
#     x,
# ):
#     del x
#     raise NotImplementedError()


# ## Cross #####################################
# @overload
# def cross(
#     a: Unitful,
#     b: Unitful,
#     *args,
#     **kwargs,
# ) -> Unitful:
#     new_val = jnp._orig_cross(a.val, b.val, *args, **kwargs)  # type: ignore
#     new_scale = a.unit.scale + b.unit.scale
#     unit_dict = dim_after_multiplication(a.unit.dim, b.unit.dim)
#     # static computation
#     new_static_arr = None
#     if is_traced(new_val):
#         x_arr = get_static_operand(a)
#         y_arr = get_static_operand(b)
#         if x_arr is not None and y_arr is not None:
#             new_static_arr = np.cross(x_arr, y_arr, *args, **kwargs)
#     return Unitful(val=new_val, unit=Unit(scale=new_scale, dim=unit_dict), static_arr=new_static_arr)


# @overload
# def cross(
#     a: jax.Array,
#     b: jax.Array,
#     *args,
#     **kwargs,
# ) -> jax.Array:
#     return jnp._orig_cross(a, b, *args, **kwargs)  # type: ignore


# @overload
# def cross(
#     a,
#     b,
#     *args,
#     **kwargs,
# ):
#     raise NotImplementedError(f"Currently not implemented for {a}, {b}, {args}, {kwargs}")


# @dispatch
# def cross(
#     a,
#     b,
#     *args,
#     **kwargs,
# ):
#     del a, b, args, kwargs
#     raise NotImplementedError()


# ## Conjugate #####################################
# @overload
# def conj(
#     x: Unitful,
# ) -> Unitful:
#     new_val = jnp._orig_conj(x.val)  # type: ignore
#     # static arr computation
#     new_static_arr = None
#     if is_traced(new_val):
#         x_arr = get_static_operand(x)
#         if x_arr is not None:
#             new_static_arr = np.conj(x_arr)
#     return Unitful(val=new_val, unit=x.unit, static_arr=new_static_arr)


# @overload
# def conj(
#     x: int,
# ) -> int:
#     return x


# @overload
# def conj(
#     x: float,
# ) -> float:
#     return x


# @overload
# def conj(
#     x: complex,
# ) -> complex:
#     return x.conjugate()


# @overload
# def conj(
#     x: jax.Array,
# ) -> jax.Array:
#     return jnp._orig_conj(x)  # type: ignore


# @dispatch
# def conj(
#     x,
# ):
#     del x
#     raise NotImplementedError()


# ## dot #####################################
# @overload
# def dot(
#     a: Unitful,
#     b: Unitful,
#     *args,
#     **kwargs,
# ) -> Unitful:
#     new_val = jnp._orig_dot(a.val, b.val, *args, **kwargs)  # type: ignore
#     unit_dict = dim_after_multiplication(a.unit.dim, b.unit.dim)
#     new_scale = a.unit.scale + b.unit.scale
#     # static computation
#     new_static_arr = None
#     if is_traced(new_val):
#         x_arr = get_static_operand(a)
#         y_arr = get_static_operand(b)
#         if x_arr is not None and y_arr is not None:
#             new_static_arr = np.dot(x_arr, y_arr, *args, **kwargs)
#     return Unitful(val=new_val, unit=Unit(scale=new_scale, dim=unit_dict), static_arr=new_static_arr)


# @overload
# def dot(
#     a: Unitful,
#     b: jax.Array | np.ndarray,
#     *args,
#     **kwargs,
# ) -> Unitful:
#     new_val = jnp._orig_dot(a.val, b, *args, **kwargs)  # type: ignore
#     # static computation
#     new_static_arr = None
#     if is_traced(new_val):
#         x_arr = get_static_operand(a)
#         y_arr = get_static_operand(b)
#         if x_arr is not None and y_arr is not None:
#             new_static_arr = np.dot(x_arr, y_arr, *args, **kwargs)
#     return Unitful(val=new_val, unit=a.unit, static_arr=new_static_arr)


# @overload
# def dot(
#     a: jax.Array | np.ndarray,
#     b: Unitful,
#     *args,
#     **kwargs,
# ) -> Unitful:
#     new_val = jnp._orig_dot(a, b.val, *args, **kwargs)  # type: ignore
#     # static computation
#     new_static_arr = None
#     if is_traced(new_val):
#         x_arr = get_static_operand(a)
#         y_arr = get_static_operand(b)
#         if x_arr is not None and y_arr is not None:
#             new_static_arr = np.dot(x_arr, y_arr, *args, **kwargs)
#     return Unitful(val=new_val, unit=b.unit, static_arr=new_static_arr)


# @overload
# def dot(
#     a: jax.Array,
#     b: jax.Array,
#     *args,
#     **kwargs,
# ) -> jax.Array:
#     return jnp._orig_dot(a, b, *args, **kwargs)  # type: ignore


# @dispatch
# def dot(
#     a,
#     b,
#     *args,
#     **kwargs,
# ):
#     del a, b, args, kwargs
#     raise NotImplementedError()


# ## Transpose #####################################
# @overload
# def transpose(
#     x: Unitful,
#     *args,
#     **kwargs,
# ) -> Unitful:
#     new_val = jnp._orig_transpose(x.val, *args, **kwargs)  # type: ignore
#     # static computation
#     new_static_arr = None
#     if is_traced(new_val):
#         x_arr = get_static_operand(x)
#         if x_arr is not None:
#             new_static_arr = np.transpose(x_arr, *args, **kwargs)
#     return Unitful(val=new_val, unit=x.unit, static_arr=new_static_arr)


# @overload
# def transpose(
#     x: jax.Array,
#     *args,
#     **kwargs,
# ) -> jax.Array:
#     return jnp._orig_transpose(x, *args, **kwargs)  # type: ignore


# @dispatch
# def transpose(
#     x,
#     *args,
#     **kwargs,
# ):
#     del x, args, kwargs
#     raise NotImplementedError()


# ## pad #####################################
# @overload
# def pad(
#     x: Unitful,
#     *args,
#     **kwargs,
# ) -> Unitful:
#     new_val = jnp._orig_pad(x.val, *args, **kwargs)  # type: ignore
#     # static computation
#     new_static_arr = None
#     if is_traced(new_val):
#         x_arr = get_static_operand(x)
#         if x_arr is not None:
#             new_static_arr = np.pad(x_arr, *args, **kwargs)
#     return Unitful(val=new_val, unit=x.unit, static_arr=new_static_arr)


# @overload
# def pad(
#     x: jax.Array,
#     *args,
#     **kwargs,
# ) -> jax.Array:
#     return jnp._orig_pad(x, *args, **kwargs)  # type: ignore


# @dispatch
# def pad(
#     x,
#     *args,
#     **kwargs,
# ):
#     del x, args, kwargs
#     raise NotImplementedError()


# ## stack #####################################
# @overload
# def stack(
#     arrays: Unitful | Sequence[Unitful],
#     *args,
#     **kwargs,
# ) -> Unitful:
#     if isinstance(arrays, Sequence):
#         for a in arrays:
#             if a.unit.dim != arrays[0].unit.dim:
#                 raise Exception(f"jnp.stack requires all Unitful to have the same dimension, but got: {arrays}")
#         # bring all values to same scale
#         new_scale, factors = handle_n_scales([a.unit.scale for a in arrays])
#         scaled = [a.val * f for a, f in zip(arrays, factors)]
#         # simply call original function
#         new_val = jnp._orig_stack(scaled, *args, **kwargs)  # type: ignore
#         new_unit = Unit(scale=new_scale, dim=arrays[0].unit.dim)
#         # static computation
#         new_static_arr = None
#         if is_traced(new_val):
#             arrs = [get_static_operand(v) for v in arrays]
#             if all([v is not None for v in arrs]):
#                 scaled_arrs = [a * f for a, f in zip(arrs, factors)]
#                 new_static_arr = np.stack(scaled_arrs, *args, **kwargs)
#     else:
#         new_val = jnp._orig_stack(arrays, *args, **kwargs)  # type: ignore
#         new_unit = arrays.unit
#         # static computation
#         new_static_arr = None
#         if is_traced(new_val):
#             x_arr = get_static_operand(arrays)
#             if x_arr is not None:
#                 new_static_arr = np.stack(x_arr, *args, **kwargs)  # ty:ignore[no-matching-overload]
#     return Unitful(val=new_val, unit=new_unit, static_arr=new_static_arr)


# @overload
# def stack(
#     x: AnyArrayLike | Sequence[AnyArrayLike],
#     *args,
#     **kwargs,
# ) -> AnyArrayLike:
#     if isinstance(x, Sequence):
#         all_physical = all([output_unitful_for_array(x_i) for x_i in x])  # ty:ignore[invalid-argument-type]
#     else:
#         all_physical = output_unitful_for_array(x)
#     # axis/dtype args/kwargs needs to be static for eval_shape
#     partial_orig_fn = jax.tree_util.Partial(jnp._orig_stack, x, *args, **kwargs)  # type: ignore
#     result_shape_dtype = jax.eval_shape(partial_orig_fn)
#     # if we cannot convert to unitful just call original function
#     if not output_unitful_for_array(result_shape_dtype) or not all_physical:
#         return jnp._orig_stack(x, *args, **kwargs)  # type: ignore
#     # convert inputs to unitful without dimension
#     if isinstance(x, Sequence):
#         unit_input = [Unitful(val=x_i, unit=EMPTY_UNIT) for x_i in x]  # type: ignore
#     else:
#         unit_input = Unitful(val=x, unit=EMPTY_UNIT)
#     unit_result = stack(unit_input, *args, **kwargs)
#     return unit_result  # type: ignore


# @dispatch
# def stack(
#     x,
#     *args,
#     **kwargs,
# ):
#     del x, args, kwargs
#     raise NotImplementedError()


# ## isfinite #####################################
# @overload
# def isfinite(
#     x: Unitful,
#     *args,
#     **kwargs,
# ) -> jax.Array:
#     new_val = jnp._orig_isfinite(x.val, *args, **kwargs)  # type: ignore
#     return new_val


# @overload
# def isfinite(
#     x: jax.Array,
#     *args,
#     **kwargs,
# ) -> jax.Array:
#     return jnp._orig_isfinite(x, *args, **kwargs)  # type: ignore


# @dispatch
# def isfinite(
#     x,
#     *args,
#     **kwargs,
# ):
#     del x, args, kwargs
#     raise NotImplementedError()


# ## real #####################################
# @overload
# def real(
#     val: Unitful,
#     *args,
#     **kwargs,
# ) -> Unitful:
#     new_val = jnp._orig_real(val.val, *args, **kwargs)  # type: ignore
#     # static computation
#     new_static_arr = None
#     if is_traced(new_val):
#         x_arr = get_static_operand(val)
#         if x_arr is not None:
#             new_static_arr = np.real(x_arr, *args, **kwargs)
#     return Unitful(val=new_val, unit=val.unit, static_arr=new_static_arr)


# @overload
# def real(val: int, *args, **kwargs) -> int:
#     return val


# @overload
# def real(val: float, *args, **kwargs) -> float:
#     return val


# @overload
# def real(val: complex, *args, **kwargs) -> complex:
#     return val.real


# @overload
# def real(val: jax.Array, *args, **kwargs) -> jax.Array:
#     return jnp._orig_real(val, *args, **kwargs)  # type: ignore


# @dispatch
# def real(
#     val,
#     *args,
#     **kwargs,
# ):
#     del val, args, kwargs
#     raise NotImplementedError()


# ## imag #####################################
# @overload
# def imag(val: Unitful, *args, **kwargs) -> Unitful:
#     new_val = jnp._orig_imag(val.val, *args, **kwargs)  # type: ignore
#     # static computation
#     new_static_arr = None
#     if is_traced(new_val):
#         x_arr = get_static_operand(val)
#         if x_arr is not None:
#             new_static_arr = np.imag(x_arr, *args, **kwargs)
#     return Unitful(val=new_val, unit=val.unit, static_arr=new_static_arr)


# @overload
# def imag(val: int, *args, **kwargs) -> int:
#     return val


# @overload
# def imag(val: float, *args, **kwargs) -> float:
#     return val


# @overload
# def imag(val: complex, *args, **kwargs) -> complex:
#     return val.imag


# @overload
# def imag(val: jax.Array, *args, **kwargs) -> jax.Array:
#     return jnp._orig_imag(val, *args, **kwargs)  # type: ignore


# @dispatch
# def imag(
#     val,
#     *args,
#     **kwargs,
# ):
#     del val, args, kwargs
#     raise NotImplementedError()


# ## sin #####################################
# @overload
# def sin(x: Unitful, *args, **kwargs) -> Unitful:
#     assert x.unit.dim == {}, (
#         f"Cannot apply trigonometric function to a unitful value with dimension {x.unit.dim}. Input must be dimensionless (radians)."
#     )

#     if (
#         isinstance(x.val, (bool, np.bool_))
#         or (isinstance(x.val, np.ndarray) and x.val.dtype == np.bool_)
#         or (isinstance(x.val, jax.Array) and x.val.dtype == jnp.bool_)
#     ):
#         raise ValueError("Function sin does not support boolean inputs.")

#     x_val_nonscale = x.val * (10**x.unit.scale)
#     if isinstance(x.val, jax.Array):
#         new_val = jnp._orig_sin(x_val_nonscale)  # type: ignore
#     elif isinstance(x.val, np.ndarray | np.number):
#         new_val = np.sin(x_val_nonscale, *args, **kwargs)
#     elif isinstance(x.val, complex):
#         new_val = cmath.sin(x_val_nonscale)
#     else:  # only int, float
#         new_val = math.sin(x_val_nonscale)

#     # static computation
#     new_static_arr = None
#     if is_traced(new_val):
#         x_arr = get_static_operand(x)
#         if x_arr is not None:
#             x_arr_nonscale = x_arr * (10**x.unit.scale)
#             new_static_arr = np.sin(x_arr_nonscale)
#     return Unitful(val=new_val, unit=EMPTY_UNIT, static_arr=new_static_arr)


# @overload
# def sin(x: int | float) -> float:
#     return math.sin(x)


# @overload
# def sin(x: complex) -> complex:
#     return cmath.sin(x)


# @overload
# def sin(x: np.ndarray | np.number, *args, **kwargs) -> np.ndarray:
#     return np.sin(x, *args, **kwargs)


# @overload
# def sin(x: jax.Array) -> jax.Array:
#     return jnp._orig_sin(x)  # type: ignore


# @dispatch
# def sin(x, *args, **kwargs):
#     del x, args, kwargs
#     raise NotImplementedError()


# ## cos #####################################
# @overload
# def cos(x: Unitful, *args, **kwargs) -> Unitful:
#     assert x.unit.dim == {}, (
#         f"Cannot apply trigonometric function to a unitful value with dimension {x.unit.dim}. Input must be dimensionless (radians)."
#     )

#     if (
#         isinstance(x.val, (bool, np.bool_))
#         or (isinstance(x.val, np.ndarray) and x.val.dtype == np.bool_)
#         or (isinstance(x.val, jax.Array) and x.val.dtype == jnp.bool_)
#     ):
#         raise ValueError("Function cos does not support boolean inputs.")

#     x_val_nonscale = x.val * (10**x.unit.scale)

#     if isinstance(x.val, jax.Array):
#         new_val = jnp._orig_cos(x_val_nonscale)  # type: ignore
#     elif isinstance(x.val, np.ndarray | np.number):
#         new_val = np.cos(x_val_nonscale, *args, **kwargs)
#     elif isinstance(x.val, complex):
#         new_val = cmath.cos(x_val_nonscale)
#     else:  # only int, float
#         new_val = math.cos(x_val_nonscale)
#     # static computation
#     new_static_arr = None
#     if is_traced(new_val):
#         x_arr = get_static_operand(x)
#         if x_arr is not None:
#             x_arr_nonscale = x_arr * (10**x.unit.scale)
#             new_static_arr = np.cos(x_arr_nonscale)
#     return Unitful(val=new_val, unit=x.unit, static_arr=new_static_arr)


# @overload
# def cos(x: jax.Array) -> jax.Array:
#     return jnp._orig_cos(x)  # type: ignore


# @overload
# def cos(x: int | float) -> float:
#     return math.cos(x)


# @overload
# def cos(x: complex) -> complex:
#     return cmath.cos(x)


# @overload
# def cos(x: np.ndarray | np.number, *args, **kwargs) -> np.ndarray:
#     return np.cos(x, *args, **kwargs)


# @dispatch
# def cos(x, *args, **kwargs):
#     del x, args, kwargs
#     raise NotImplementedError()


# ## tan #####################################
# @overload
# def tan(x: Unitful, *args, **kwargs) -> Unitful:
#     assert x.unit.dim == {}, (
#         f"Cannot apply trigonometric function to a unitful value with dimension {x.unit.dim}. Input must be dimensionless (radians)."
#     )

#     if (
#         isinstance(x.val, (bool, np.bool_))
#         or (isinstance(x.val, np.ndarray) and x.val.dtype == np.bool_)
#         or (isinstance(x.val, jax.Array) and x.val.dtype == jnp.bool_)
#     ):
#         raise ValueError("Function tan does not support boolean inputs.")

#     x_val_nonscale = x.val * (10**x.unit.scale)
#     if isinstance(x.val, jax.Array):
#         new_val = jnp._orig_tan(x_val_nonscale)  # type: ignore
#     elif isinstance(x.val, np.ndarray | np.number):
#         new_val = np.tan(x_val_nonscale, *args, **kwargs)
#     elif isinstance(x.val, complex):
#         new_val = cmath.tan(x_val_nonscale)
#     else:  # only int, float
#         new_val = math.tan(x_val_nonscale)
#     # static computation
#     new_static_arr = None
#     if is_traced(new_val):
#         x_arr = get_static_operand(x)
#         if x_arr is not None:
#             x_arr_nonscale = x_arr * (10**x.unit.scale)
#             new_static_arr = np.tan(x_arr_nonscale)
#     return Unitful(val=new_val, unit=x.unit, static_arr=new_static_arr)


# @overload
# def tan(x: jax.Array) -> jax.Array:
#     return jnp._orig_tan(x)  # type: ignore


# @overload
# def tan(x: int | float) -> float:
#     return math.tan(x)


# @overload
# def tan(x: complex) -> complex:
#     return cmath.tan(x)


# @overload
# def tan(x: np.ndarray | np.number, *args, **kwargs) -> np.ndarray:
#     return np.tan(x, *args, **kwargs)


# @dispatch
# def tan(x, *args, **kwargs):
#     del x, args, kwargs
#     raise NotImplementedError()


# ## arcsin #######################################
# @overload
# def arcsin(x: Unitful, *args, **kwargs) -> Unitful:
#     assert x.unit.dim == {}, (
#         f"Cannot apply trigonometric function to a unitful value with dimension {x.unit.dim}. Input must be dimensionless (radians)."
#     )

#     if (
#         isinstance(x.val, (bool, np.bool_))
#         or (isinstance(x.val, np.ndarray) and x.val.dtype == np.bool_)
#         or (isinstance(x.val, jax.Array) and x.val.dtype == jnp.bool_)
#     ):
#         raise ValueError("Function arcsin does not support boolean inputs.")

#     x_val_nonscale = x.val * (10**x.unit.scale)
#     if isinstance(x.val, jax.Array):
#         new_val = jnp._orig_arcsin(x_val_nonscale)  # type: ignore
#     elif isinstance(x.val, np.ndarray | np.number):
#         new_val = np.arcsin(x_val_nonscale, *args, **kwargs)
#     elif isinstance(x.val, complex):
#         new_val = cmath.asin(x_val_nonscale)
#     else:  # only int, float
#         new_val = math.asin(x_val_nonscale)
#     # static computation
#     new_static_arr = None
#     if is_traced(new_val):
#         x_arr = get_static_operand(x)
#         if x_arr is not None:
#             x_arr_nonscale = x_arr * (10**x.unit.scale)
#             new_static_arr = np.arcsin(x_arr_nonscale)
#     return Unitful(val=new_val, unit=x.unit, static_arr=new_static_arr)


# @overload
# def arcsin(x: int | float) -> float:
#     return math.asin(x)


# @overload
# def arcsin(x: complex) -> complex:
#     return cmath.asin(x)


# @overload
# def arcsin(x: jax.Array) -> jax.Array:
#     return jnp._orig_arcsin(x)  # type: ignore


# @overload
# def arcsin(x: np.ndarray, *args, **kwargs) -> np.ndarray:
#     return np.arcsin(x, *args, **kwargs)


# @dispatch
# def arcsin(x, *args, **kwargs):
#     del x, args, kwargs
#     raise NotImplementedError()


# ## arccos #######################################
# @overload
# def arccos(x: Unitful, *args, **kwargs) -> Unitful:
#     assert x.unit.dim == {}, (
#         f"Cannot apply trigonometric function to a unitful value with dimension {x.unit.dim}. Input must be dimensionless (radians)."
#     )

#     if (
#         isinstance(x.val, (bool, np.bool_))
#         or (isinstance(x.val, np.ndarray) and x.val.dtype == np.bool_)
#         or (isinstance(x.val, jax.Array) and x.val.dtype == jnp.bool_)
#     ):
#         raise ValueError("Function arccos does not support boolean inputs.")

#     x_val_nonscale = x.val * (10**x.unit.scale)
#     if isinstance(x.val, jax.Array):
#         new_val = jnp._orig_arccos(x_val_nonscale)  # type: ignore
#     elif isinstance(x.val, np.ndarray | np.number):
#         new_val = np.arccos(x_val_nonscale, *args, **kwargs)
#     elif isinstance(x.val, complex):
#         new_val = cmath.acos(x_val_nonscale)
#     else:  # only int, float
#         new_val = math.acos(x_val_nonscale)
#     # static computation
#     new_static_arr = None
#     if is_traced(new_val):
#         x_arr = get_static_operand(x)
#         if x_arr is not None:
#             x_arr_nonscale = x_arr * (10**x.unit.scale)
#             new_static_arr = np.arccos(x_arr_nonscale)
#     return Unitful(val=new_val, unit=x.unit, static_arr=new_static_arr)


# @overload
# def arccos(x: int | float) -> float:
#     return math.acos(x)


# @overload
# def arccos(x: complex) -> complex:
#     return cmath.acos(x)


# @overload
# def arccos(x: jax.Array) -> jax.Array:
#     return jnp._orig_arccos(x)  # type: ignore


# @overload
# def arccos(x: np.ndarray, *args, **kwargs) -> np.ndarray:
#     return np.arccos(x, *args, **kwargs)


# @dispatch
# def arccos(x, *args, **kwargs):
#     del x, args, kwargs
#     raise NotImplementedError()


# ## arctan #######################################
# @overload
# def arctan(x: Unitful, *args, **kwargs) -> Unitful:
#     assert x.unit.dim == {}, (
#         f"Cannot apply trigonometric function to a unitful value with dimension {x.unit.dim}. Input must be dimensionless (radians)."
#     )

#     if (
#         isinstance(x.val, (bool, np.bool_))
#         or (isinstance(x.val, np.ndarray) and x.val.dtype == np.bool_)
#         or (isinstance(x.val, jax.Array) and x.val.dtype == jnp.bool_)
#     ):
#         raise ValueError("Function arctan does not support boolean inputs.")

#     x_val_nonscale = x.val * (10**x.unit.scale)
#     if isinstance(x.val, jax.Array):
#         new_val = jnp._orig_arctan(x_val_nonscale)  # type: ignore
#     elif isinstance(x.val, np.ndarray | np.number):
#         new_val = np.arctan(x_val_nonscale, *args, **kwargs)
#     elif isinstance(x.val, complex):
#         new_val = cmath.atan(x_val_nonscale)
#     else:  # only int, float
#         new_val = math.atan(x_val_nonscale)
#     # static computation
#     new_static_arr = None
#     if is_traced(new_val):
#         x_arr = get_static_operand(x)
#         if x_arr is not None:
#             x_arr_nonscale = x_arr * (10**x.unit.scale)
#             new_static_arr = np.arctan(x_arr_nonscale)
#     return Unitful(val=new_val, unit=x.unit, static_arr=new_static_arr)


# @overload
# def arctan(x: int | float) -> float:
#     return math.atan(x)


# @overload
# def arctan(x: complex) -> complex:
#     return cmath.atan(x)


# @overload
# def arctan(x: jax.Array) -> jax.Array:
#     return jnp._orig_arctan(x)  # type: ignore


# @overload
# def arctan(x: np.ndarray, *args, **kwargs) -> np.ndarray:
#     return np.arctan(x, *args, **kwargs)


# @dispatch
# def arctan(x, *args, **kwargs):
#     del x, args, kwargs
#     raise NotImplementedError()


# ## sinh #######################################
# @overload
# def sinh(x: Unitful, *args, **kwargs) -> Unitful:
#     assert x.unit.dim == {}, (
#         f"Cannot apply trigonometric function to a unitful value with dimension {x.unit.dim}. Input must be dimensionless (radians)."
#     )

#     if (
#         isinstance(x.val, (bool, np.bool_))
#         or (isinstance(x.val, np.ndarray) and x.val.dtype == np.bool_)
#         or (isinstance(x.val, jax.Array) and x.val.dtype == jnp.bool_)
#     ):
#         raise ValueError("Function sinh does not support boolean inputs.")

#     x_val_nonscale = x.val * (10**x.unit.scale)
#     if isinstance(x.val, jax.Array):
#         new_val = jnp._orig_sinh(x_val_nonscale)  # type: ignore
#     elif isinstance(x.val, np.ndarray | np.number):
#         new_val = np.sinh(x_val_nonscale, *args, **kwargs)
#     elif isinstance(x.val, complex):
#         new_val = cmath.sinh(x_val_nonscale)
#     else:  # only int, float
#         new_val = math.sinh(x_val_nonscale)
#     # static computation
#     new_static_arr = None
#     if is_traced(new_val):
#         x_arr = get_static_operand(x)
#         if x_arr is not None:
#             x_arr_nonscale = x_arr * (10**x.unit.scale)
#             new_static_arr = np.sinh(x_arr_nonscale)
#     return Unitful(val=new_val, unit=x.unit, static_arr=new_static_arr)


# @overload
# def sinh(x: int | float) -> float:
#     return math.sinh(x)


# @overload
# def sinh(x: complex) -> complex:
#     return cmath.sinh(x)


# @overload
# def sinh(x: jax.Array) -> jax.Array:
#     return jnp._orig_sinh(x)  # type: ignore


# @overload
# def sinh(x: np.ndarray, *args, **kwargs) -> np.ndarray:
#     return np.sinh(x, *args, **kwargs)


# @dispatch
# def sinh(x, *args, **kwargs):
#     del x, args, kwargs
#     raise NotImplementedError()


# ## cosh #######################################
# @overload
# def cosh(x: Unitful, *args, **kwargs) -> Unitful:
#     assert x.unit.dim == {}, (
#         f"Cannot apply trigonometric function to a unitful value with dimension {x.unit.dim}. Input must be dimensionless (radians)."
#     )

#     if (
#         isinstance(x.val, (bool, np.bool_))
#         or (isinstance(x.val, np.ndarray) and x.val.dtype == np.bool_)
#         or (isinstance(x.val, jax.Array) and x.val.dtype == jnp.bool_)
#     ):
#         raise ValueError("Function cosh does not support boolean inputs.")

#     x_val_nonscale = x.val * (10**x.unit.scale)
#     if isinstance(x.val, jax.Array):
#         new_val = jnp._orig_cosh(x_val_nonscale)  # type: ignore
#     elif isinstance(x.val, np.ndarray | np.number):
#         new_val = np.cosh(x_val_nonscale, *args, **kwargs)
#     elif isinstance(x.val, complex):
#         new_val = cmath.cosh(x_val_nonscale)
#     else:  # only int, float
#         new_val = math.cosh(x_val_nonscale)
#     # static computation
#     new_static_arr = None
#     if is_traced(new_val):
#         x_arr = get_static_operand(x)
#         if x_arr is not None:
#             x_arr_nonscale = x_arr * (10**x.unit.scale)
#             new_static_arr = np.cosh(x_arr_nonscale)
#     return Unitful(val=new_val, unit=x.unit, static_arr=new_static_arr)


# @overload
# def cosh(x: int | float) -> float:
#     return math.cosh(x)


# @overload
# def cosh(x: complex) -> complex:
#     return cmath.cosh(x)


# @overload
# def cosh(x: jax.Array) -> jax.Array:
#     return jnp._orig_cosh(x)  # type: ignore


# @overload
# def cosh(x: np.ndarray, *args, **kwargs) -> np.ndarray:
#     return np.cosh(x, *args, **kwargs)


# @dispatch
# def cosh(x, *args, **kwargs):
#     del x, args, kwargs
#     raise NotImplementedError()


# ## tanh #######################################
# @overload
# def tanh(x: Unitful, *args, **kwargs) -> Unitful:
#     assert x.unit.dim == {}, (
#         f"Cannot apply trigonometric function to a unitful value with dimension {x.unit.dim}. Input must be dimensionless (radians)."
#     )

#     if (
#         isinstance(x.val, (bool, np.bool_))
#         or (isinstance(x.val, np.ndarray) and x.val.dtype == np.bool_)
#         or (isinstance(x.val, jax.Array) and x.val.dtype == jnp.bool_)
#     ):
#         raise ValueError("Function tanh does not support boolean inputs.")

#     x_val_nonscale = x.val * (10**x.unit.scale)
#     if isinstance(x.val, jax.Array):
#         new_val = jnp._orig_tanh(x_val_nonscale)  # type: ignore
#     elif isinstance(x.val, np.ndarray | np.number):
#         new_val = np.tanh(x_val_nonscale, *args, **kwargs)
#     elif isinstance(x.val, complex):
#         new_val = cmath.tanh(x_val_nonscale)
#     else:  # only int, float
#         new_val = math.tanh(x_val_nonscale)
#     # static computation
#     new_static_arr = None
#     if is_traced(new_val):
#         x_arr = get_static_operand(x)
#         if x_arr is not None:
#             x_arr_nonscale = x_arr * (10**x.unit.scale)
#             new_static_arr = np.tanh(x_arr_nonscale)
#     return Unitful(val=new_val, unit=x.unit, static_arr=new_static_arr)


# @overload
# def tanh(x: int | float) -> float:
#     return math.tanh(x)


# @overload
# def tanh(x: complex) -> complex:
#     return cmath.tanh(x)


# @overload
# def tanh(x: jax.Array) -> jax.Array:
#     return jnp._orig_tanh(x)  # type: ignore


# @overload
# def tanh(x: np.ndarray, *args, **kwargs) -> np.ndarray:
#     return np.tanh(x, *args, **kwargs)


# @dispatch
# def tanh(x, *args, **kwargs):
#     del x, args, kwargs
#     raise NotImplementedError()


# ## arcsinh #######################################
# @overload
# def arcsinh(x: Unitful, *args, **kwargs) -> Unitful:
#     assert x.unit.dim == {}, (
#         f"Cannot apply trigonometric function to a unitful value with dimension {x.unit.dim}. Input must be dimensionless (radians)."
#     )

#     if (
#         isinstance(x.val, (bool, np.bool_))
#         or (isinstance(x.val, np.ndarray) and x.val.dtype == np.bool_)
#         or (isinstance(x.val, jax.Array) and x.val.dtype == jnp.bool_)
#     ):
#         raise ValueError("Function arcsinh does not support boolean inputs.")

#     x_val_nonscale = x.val * (10**x.unit.scale)
#     if isinstance(x.val, jax.Array):
#         new_val = jnp._orig_arcsinh(x_val_nonscale)  # type: ignore
#     elif isinstance(x.val, np.ndarray | np.number):
#         new_val = np.arcsinh(x_val_nonscale, *args, **kwargs)
#     elif isinstance(x.val, complex):
#         new_val = cmath.asinh(x_val_nonscale)
#     else:  # only int, float
#         new_val = math.asinh(x_val_nonscale)
#     # static computation
#     new_static_arr = None
#     if is_traced(new_val):
#         x_arr = get_static_operand(x)
#         if x_arr is not None:
#             x_arr_nonscale = x_arr * (10**x.unit.scale)
#             new_static_arr = np.arcsinh(x_arr_nonscale)
#     return Unitful(val=new_val, unit=x.unit, static_arr=new_static_arr)


# @overload
# def arcsinh(x: int | float) -> float:
#     return math.asinh(x)


# @overload
# def arcsinh(x: complex) -> complex:
#     return cmath.asinh(x)


# @overload
# def arcsinh(x: jax.Array) -> jax.Array:
#     return jnp._orig_arcsinh(x)  # type: ignore


# @overload
# def arcsinh(x: np.ndarray, *args, **kwargs) -> np.ndarray:
#     return np.arcsinh(x, *args, **kwargs)


# @dispatch
# def arcsinh(x, *args, **kwargs):
#     del x, args, kwargs
#     raise NotImplementedError()


# ## arccosh #######################################
# @overload
# def arccosh(x: Unitful, *args, **kwargs) -> Unitful:
#     assert x.unit.dim == {}, (
#         f"Cannot apply trigonometric function to a unitful value with dimension {x.unit.dim}. Input must be dimensionless (radians)."
#     )

#     if (
#         isinstance(x.val, (bool, np.bool_))
#         or (isinstance(x.val, np.ndarray) and x.val.dtype == np.bool_)
#         or (isinstance(x.val, jax.Array) and x.val.dtype == jnp.bool_)
#     ):
#         raise ValueError("Function arccosh does not support boolean inputs.")

#     x_val_nonscale = x.val * (10**x.unit.scale)
#     if isinstance(x.val, jax.Array):
#         new_val = jnp._orig_arccosh(x_val_nonscale)  # type: ignore
#     elif isinstance(x.val, np.ndarray | np.number):
#         new_val = np.arccosh(x_val_nonscale, *args, **kwargs)
#     elif isinstance(x.val, complex):
#         new_val = cmath.acosh(x_val_nonscale)
#     else:  # only int, float
#         new_val = math.acosh(x_val_nonscale)
#     # static computation
#     new_static_arr = None
#     if is_traced(new_val):
#         x_arr = get_static_operand(x)
#         if x_arr is not None:
#             x_arr_nonscale = x_arr * (10**x.unit.scale)
#             new_static_arr = np.arccosh(x_arr_nonscale)
#     return Unitful(val=new_val, unit=x.unit, static_arr=new_static_arr)


# @overload
# def arccosh(x: int | float) -> float:
#     return math.acosh(x)


# @overload
# def arccosh(x: complex) -> complex:
#     return cmath.acosh(x)


# @overload
# def arccosh(x: jax.Array) -> jax.Array:
#     return jnp._orig_arccosh(x)  # type: ignore


# @overload
# def arccosh(x: np.ndarray, *args, **kwargs) -> np.ndarray:
#     return np.arccosh(x, *args, **kwargs)


# @dispatch
# def arccosh(x, *args, **kwargs):
#     del x, args, kwargs
#     raise NotImplementedError()


# ## arctanh #######################################
# @overload
# def arctanh(x: Unitful, *args, **kwargs) -> Unitful:
#     assert x.unit.dim == {}, (
#         f"Cannot apply trigonometric function to a unitful value with dimension {x.unit.dim}. Input must be dimensionless (radians)."
#     )

#     if (
#         isinstance(x.val, (bool, np.bool_))
#         or (isinstance(x.val, np.ndarray) and x.val.dtype == np.bool_)
#         or (isinstance(x.val, jax.Array) and x.val.dtype == jnp.bool_)
#     ):
#         raise ValueError("Function arctanh does not support boolean inputs.")

#     x_val_nonscale = x.val * (10**x.unit.scale)
#     if isinstance(x.val, jax.Array):
#         new_val = jnp._orig_arctanh(x_val_nonscale)  # type: ignore
#     elif isinstance(x.val, np.ndarray | np.number):
#         new_val = np.arctanh(x_val_nonscale, *args, **kwargs)
#     elif isinstance(x.val, complex):
#         new_val = cmath.atanh(x_val_nonscale)
#     else:  # only int, float
#         new_val = math.atanh(x_val_nonscale)
#     # static computation
#     new_static_arr = None
#     if is_traced(new_val):
#         x_arr = get_static_operand(x)
#         if x_arr is not None:
#             x_arr_nonscale = x_arr * (10**x.unit.scale)
#             new_static_arr = np.arctanh(x_arr_nonscale)
#     return Unitful(val=new_val, unit=x.unit, static_arr=new_static_arr)


# @overload
# def arctanh(x: int | float) -> float:
#     return math.atanh(x)


# @overload
# def arctanh(x: complex) -> complex:
#     return cmath.atanh(x)


# @overload
# def arctanh(x: jax.Array) -> jax.Array:
#     return jnp._orig_arctanh(x)  # type: ignore


# @overload
# def arctanh(x: np.ndarray, *args, **kwargs) -> np.ndarray:
#     return np.arctanh(x, *args, **kwargs)


# @dispatch
# def arctanh(x, *args, **kwargs):
#     del x, args, kwargs
#     raise NotImplementedError()


# ## asarray #####################################
# def asarray(
#     a,
#     *args,
#     **kwargs,
# ) -> jax.Array:
#     materialised = jax.tree.map(
#         lambda x: x.materialise() if isinstance(x, Unitful) else x,
#         a,
#         is_leaf=lambda x: isinstance(x, Unitful),
#     )
#     partial_orig_fn = jax.tree_util.Partial(jnp._orig_asarray, materialised, *args, **kwargs)  # type: ignore
#     result_shape_dtype = jax.eval_shape(partial_orig_fn)
#     result: jax.Array = jnp._orig_asarray(materialised, *args, **kwargs)  # type: ignore
#     if not output_unitful_for_array(result_shape_dtype):
#         # cannot use this as Unitful, wrong dtype
#         return result

#     # try to get a static version of the array and save to trace metadata
#     static_arr = None
#     result_size = math.prod(result.shape)
#     if is_struct_optimizable(a) and result_size <= MAX_STATIC_OPTIMIZED_SIZE:
#         static_arr = np.asarray(a, copy=True)

#     # return Unitful without unit. We lie to typechecker here
#     return Unitful(val=result, unit=Unit(scale=0, dim={}), static_arr=static_arr)  # type: ignore


# def array(
#     a,
#     *args,
#     **kwargs,
# ) -> jax.Array:
#     return asarray(a, *args, **kwargs)


# ## exp #########################################
# @overload
# def exp(
#     x: Unitful,
# ) -> Unitful:
#     assert x.unit.dim == {}, f"Cannot use unitful with dim {x.unit.dim} as exponent"
#     # TODO: improve numerical accuracy
#     new_val = jnp._orig_exp((10**x.unit.scale) * x.val)  # type: ignore
#     # static arr
#     new_static_arr = None
#     if is_traced(new_val):
#         x_arr = get_static_operand(x)
#         if x_arr is not None:
#             new_static_arr = np.exp((10**x.unit.scale) * x_arr)
#     return Unitful(val=new_val, unit=Unit(scale=0, dim={}), static_arr=new_static_arr)


# @overload
# def exp(
#     x: jax.Array,
# ) -> jax.Array:
#     return jnp._orig_exp(x)  # type: ignore


# @overload
# def exp(x: int | float) -> float:
#     return math.exp(x)


# @dispatch
# def exp(
#     x,
# ):
#     del x
#     raise NotImplementedError()


# ## expand dims #########################################
# @overload
# def expand_dims(
#     x: Unitful,
#     *args,
#     **kwargs,
# ) -> Unitful:
#     new_val = jnp._orig_expand_dims(x.val, *args, **kwargs)  # type: ignore
#     # static arr
#     new_static_arr = None
#     if is_traced(new_val):
#         x_arr = get_static_operand(x)
#         if x_arr is not None:
#             new_static_arr = np.expand_dims(x_arr, *args, **kwargs)
#     return Unitful(val=new_val, unit=x.unit, static_arr=new_static_arr)


# @overload
# def expand_dims(
#     x: jax.Array,
#     *args,
#     **kwargs,
# ) -> jax.Array:
#     return jnp._orig_expand_dims(x, *args, **kwargs)  # type: ignore


# @dispatch
# def expand_dims(x, *args, **kwargs):
#     del x, args, kwargs
#     raise NotImplementedError()


# ## where #########################################
# @overload
# def where(
#     condition: Unitful,
#     x: Unitful,
#     y: Unitful,
#     *args,
#     **kwargs,
# ) -> Unitful:
#     assert x.unit.dim == y.unit.dim, f"{x} and {y} need the same units for jnp.where"
#     assert condition.unit.dim == {} and condition.unit.scale == 0, f"Invalid condition input: {condition}"
#     x_align, y_align = align_scales(x, y)
#     c_val = condition.val

#     if any(isinstance(v, jax.Array) for v in [x_align.val, y_align.val, c_val]):
#         new_val = jnp._orig_where(c_val, x_align.val, y_align.val, *args, **kwargs)  # type: ignore
#     elif any(isinstance(v, np.ndarray) for v in (x_align.val, y_align.val, c_val)):
#         new_val = np.where(c_val, x_align.val, y_align.val, *args, **kwargs)
#     else:
#         try:  # python scalar
#             new_val = np.where(c_val, x_align.val, y_align.val, *args, **kwargs)
#         except Exception as e:
#             raise TypeError(
#                 f"Invalid input types for where(): {type(c_val)}, {type(x_align.val)}, {type(y_align.val)}"
#             ) from e

#     # static arr
#     new_static_arr = None
#     if is_traced(new_val):
#         x_arr = get_static_operand(x)
#         y_arr = get_static_operand(y)
#         c_arr = get_static_operand(condition)
#         if x_arr is not None and y_arr is not None and c_arr is not None:
#             new_static_arr = np.where(c_arr, x_arr, y_arr, *args, **kwargs)
#     return Unitful(val=new_val, unit=x_align.unit, static_arr=new_static_arr)


# @overload
# def where(
#     condition: AnyArrayLike | Unitful,
#     x: AnyArrayLike | Unitful,
#     y: Unitful,
#     *args,
#     **kwargs,
# ) -> Unitful:
#     if isinstance(condition, Unitful) and isinstance(x, Unitful):
#         return where(condition, x, y, *args, **kwargs)

#     c = condition if isinstance(condition, Unitful) else Unitful(val=condition, unit=EMPTY_UNIT)
#     new_x = x if isinstance(x, Unitful) else Unitful(val=x, unit=EMPTY_UNIT)

#     return where(c, new_x, y, *args, **kwargs)


# @overload
# def where(
#     condition: AnyArrayLike | Unitful,
#     x: Unitful,
#     y: AnyArrayLike | Unitful,
#     *args,
#     **kwargs,
# ) -> Unitful:
#     if isinstance(y, Unitful) and isinstance(condition, Unitful):
#         return where(condition, x, y, *args, **kwargs)

#     c = condition if isinstance(condition, Unitful) else Unitful(val=condition, unit=EMPTY_UNIT)
#     new_y = y if isinstance(y, Unitful) else Unitful(val=y, unit=EMPTY_UNIT)

#     return where(c, x, new_y, *args, **kwargs)


# @overload
# def where(
#     condition: jax.Array,
#     x: jax.Array,
#     y: jax.Array,
#     *args,
#     **kwargs,
# ) -> jax.Array:
#     return jnp._orig_where(condition, x, y, *args, **kwargs)  # type: ignore


# @overload
# def where(
#     condition: jax.Array,
#     x: AnyArrayLike,
#     y: AnyArrayLike,
#     *args,
#     **kwargs,
# ) -> jax.Array:
#     x = x if isinstance(x, jax.Array) else jnp.asarray(x)
#     y = y if isinstance(y, jax.Array) else jnp.asarray(y)
#     return jnp._orig_where(condition, x, y, *args, **kwargs)  # type: ignore


# @overload
# def where(
#     condition: AnyArrayLike,
#     x: jax.Array,
#     y: AnyArrayLike,
#     *args,
#     **kwargs,
# ) -> jax.Array:
#     c = condition if isinstance(condition, jax.Array) else jnp.asarray(condition)
#     y = y if isinstance(y, jax.Array) else jnp.asarray(y)
#     return jnp._orig_where(c, x, y, *args, **kwargs)  # type: ignore


# @overload
# def where(
#     condition: AnyArrayLike,
#     x: AnyArrayLike,
#     y: jax.Array,
#     *args,
#     **kwargs,
# ) -> jax.Array:
#     c = condition if isinstance(condition, jax.Array) else jnp.asarray(condition)
#     x = x if isinstance(x, jax.Array) else jnp.asarray(x)
#     return jnp._orig_where(c, x, y, *args, **kwargs)  # type: ignore


# @overload
# def where(
#     condition: np.ndarray | np.bool_ | np.number | bool | int | float | complex,
#     x: np.ndarray | np.bool_ | np.number | bool | int | float | complex,
#     y: np.ndarray | np.bool_ | np.number | bool | int | float | complex,
#     *args,
#     **kwargs,
# ) -> np.ndarray:
#     # typing excludes JAX, runtime converts any JAX inputs to NumPy.

#     c = condition if isinstance(condition, np.ndarray | np.number) else np.asarray(condition)
#     x = x if isinstance(x, np.ndarray | np.number) else np.asarray(x)
#     y = y if isinstance(y, np.ndarray | np.number) else np.asarray(y)

#     return np.where(c, x, y, *args, **kwargs)


# @dispatch
# def where(condition, x, y, *args, **kwargs):
#     del condition, x, y, args, kwargs
#     raise NotImplementedError()


# ## arange #########################################
# def arange(
#     *args,
#     **kwargs,
# ) -> Unitful:
#     # test if we should output unitful instead of array
#     partial_orig_fn = jax.tree_util.Partial(jnp._orig_arange, *args, **kwargs)  # type: ignore
#     result_shape_dtype = jax.eval_shape(partial_orig_fn)
#     result = jnp._orig_arange(*args, **kwargs)  # type: ignore
#     if not output_unitful_for_array(result_shape_dtype):
#         return result
#     # return Unitful instead of array
#     static_arr = np.arange(*args, **kwargs)
#     return Unitful(val=result, unit=EMPTY_UNIT, static_arr=static_arr)


# ## meshgrid ########################################
# @overload
# def meshgrid(
#     *args: Unitful,
#     **kwargs,
# ) -> list[Unitful]:
#     dim = args[0].unit.dim
#     for xi in args:
#         if xi.unit.dim != dim:
#             raise Exception(f"Inconsistent units in meshgrid: {args}")
#     # align the scales
#     orig_scales = [a.unit.scale for a in args]
#     new_scale, factors = handle_n_scales(orig_scales)
#     aligned_arrs = [a.val * f for a, f in zip(args, factors)]
#     new_vals = jnp._orig_meshgrid(*aligned_arrs, **kwargs)  # type: ignore
#     # test if we should create static arr as well
#     static_ops = [get_static_operand(a) for a in args]
#     if all([can_perform_static_ops(o) for o in static_ops]):
#         scaled_ops = [s * f for s, f in zip(static_ops, factors)]
#         static_arrs = np.meshgrid(*scaled_ops, **kwargs)
#         new_unitfuls = [
#             Unitful(val=v, unit=Unit(scale=new_scale, dim=dim), static_arr=s) for v, s in zip(new_vals, static_arrs)
#         ]
#     else:
#         new_unitfuls = [Unitful(val=v, unit=Unit(scale=new_scale, dim=dim)) for v in new_vals]
#     return new_unitfuls


# @overload
# def meshgrid(
#     *args: AnyArrayLike,
#     **kwargs,
# ) -> list[AnyArrayLike]:
#     # TODO: return Unitfuls instead of arrays if situation allows it
#     return jnp._orig_meshgrid(*args, **kwargs)  # type: ignore


# @dispatch
# def meshgrid(
#     *args,
#     **kwargs,
# ):
#     del args, kwargs
#     raise NotImplementedError()


# ## floor #########################################
# @overload
# def floor(
#     x: Unitful,
#     *args,
#     **kwargs,
# ) -> Unitful:
#     assert x.unit.dim == {}, f"Cannot use floor on a value with non-empty unit: {x}"
#     new_val = jnp._orig_floor(x.materialise(), *args, **kwargs)  # type: ignore
#     # static arr
#     new_static_arr = None
#     if is_traced(new_val):
#         x_arr = get_static_operand(x)
#         if x_arr is not None:
#             new_static_arr = np.floor(x_arr * (10**x.unit.scale), *args, **kwargs)
#     return Unitful(val=new_val, unit=x.unit, static_arr=new_static_arr)


# @overload
# def floor(
#     x: jax.Array,
#     *args,
#     **kwargs,
# ) -> jax.Array:
#     return jnp._orig_floor(x, *args, **kwargs)  # type: ignore


# @dispatch
# def floor(x, *args, **kwargs):
#     del x, args, kwargs
#     raise NotImplementedError()


# ## ceil #########################################
# @overload
# def ceil(
#     x: Unitful,
#     *args,
#     **kwargs,
# ) -> Unitful:
#     assert x.unit.dim == {}, f"Cannot use floor on a value with non-empty unit: {x}"
#     new_val = jnp._orig_ceil(x.materialise(), *args, **kwargs)  # type: ignore
#     # static arr
#     new_static_arr = None
#     if is_traced(new_val):
#         x_arr = get_static_operand(x)
#         if x_arr is not None:
#             new_static_arr = np.ceil(x_arr * (10**x.unit.scale), *args, **kwargs)
#     return Unitful(val=new_val, unit=x.unit, static_arr=new_static_arr)


# @overload
# def ceil(
#     x: jax.Array,
#     *args,
#     **kwargs,
# ) -> jax.Array:
#     return jnp._orig_ceil(x, *args, **kwargs)  # type: ignore


# @dispatch
# def ceil(x, *args, **kwargs):
#     del x, args, kwargs
#     raise NotImplementedError()
