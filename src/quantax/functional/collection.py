from __future__ import annotations
from quantax.functional.numpy.comparisons import le, lt, ge, gt, eq, ne, array_equal, get_array_equal_original
from quantax.functional.numpy.comparisons import constraints_comparison, get_le_original, get_lt_original, get_ge_original, get_gt_original, get_eq_original, get_ne_original

from typing import Callable

from quantax.functional.artificial import constraints_noop, noop
from quantax.functional.numpy.basic import (
    add,
    constraints_add_sub,
    constraints_divide,
    constraints_multiply,
    divide,
    get_add_original,
    get_divide_original,
    get_multiply_original,
    get_subtract_original,
    multiply,
    subtract,
)

CONSTRAINTS_DICT: dict[str, Callable] = {
    "multiply": constraints_multiply,
    "add": constraints_add_sub,
    "subtract": constraints_add_sub,
    "divide": constraints_divide,
    "noop": constraints_noop,
    "le": constraints_comparison,
    "lt": constraints_comparison,
    "ge": constraints_comparison,
    "gt": constraints_comparison,
    "eq": constraints_comparison,
    "ne": constraints_comparison,
    "array_equal": constraints_comparison,
}

FUNCTION_DICT: dict[str, Callable] = {
    "multiply": multiply,
    "add": add,
    "subtract": subtract,
    "divide": divide,
    "noop": noop,
    "le": le,
    "lt": lt,
    "ge": ge, 
    "gt": gt, 
    "eq": eq, 
    "ne": ne,
    "array_equal": array_equal,
}

ORIG_FUNCTION_DICT: dict[str, Callable] = {
    "multiply": get_multiply_original(),
    "add": get_add_original(),
    "subtract": get_subtract_original(),
    "divide": get_divide_original(),
    "le": get_le_original(),
    "lt": get_lt_original(),
    "ge": get_ge_original(),
    "gt": get_gt_original(),
    "eq": get_eq_original(),
    "ne": get_ne_original(),
    "array_equal": get_array_equal_original(),
}
