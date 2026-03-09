from __future__ import annotations

from typing import Callable

from quantax.functional.artificial import constraints_noop, noop
from quantax.functional.numpy.basic import constraints_multiply, get_multiply_original, multiply

CONSTRAINTS_DICT: dict[str, Callable] = {
    "multiply": constraints_multiply,
    "noop": constraints_noop,
}

FUNCTION_DICT: dict[str, Callable] = {
    "multiply": multiply,
    "noop": noop,
}

ORIG_FUNCTION_DICT: dict[str, Callable] = {
    "multiply": get_multiply_original(),
}
