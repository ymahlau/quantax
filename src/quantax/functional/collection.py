from __future__ import annotations
from typing import Callable
from quantax.functional.numpy import constraints_multiply, multiply


CONSTRAINTS_DICT: dict[str, Callable] = {
    "multiply": constraints_multiply,
}

FUNCTION_DICT: dict[str, Callable] = {
    "multiply": multiply,
}

