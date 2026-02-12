from typing import Callable
from quantax.functional.numpy import constraints_multiply


CONSTRAINTS_DICT: dict[str, Callable] = {
    "multiply": constraints_multiply,
}
