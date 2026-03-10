from __future__ import annotations

from typing import Callable

from quantax.functional.artificial import constraints_noop, noop
from quantax.functional.numpy.basic import (
    constraints_multiply, get_multiply_original, multiply,
    constraints_add_sub, get_add_original, add,
    get_subtract_original, subtract,
    constraints_divide, get_divide_original, divide,
)

CONSTRAINTS_DICT: dict[str, Callable] = {
    "multiply": constraints_multiply,
    "add": constraints_add_sub,
    "subtract": constraints_add_sub,
    "divide": constraints_divide,
    "noop": constraints_noop,
}

FUNCTION_DICT: dict[str, Callable] = {
    "multiply": multiply,
    "add": add,
    "subtract": subtract,
    "divide": divide,
    "noop": noop,
}

ORIG_FUNCTION_DICT: dict[str, Callable] = {
    "multiply": get_multiply_original(),
    "add": get_add_original(),
    "subtract": get_subtract_original(),
    "divide": get_divide_original(),
}
