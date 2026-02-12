from typing import Union
import jax
import numpy as np

from quantax.core.constants import MAX_STATIC_OPTIMIZED_SIZE
from quantax.core.glob import STATIC_OPTIM_STOP_FLAG
from quantax.core.typing import AnyArrayLike, StaticArrayLike
from quantax.core.utils import is_traced
from quantax.unitful.tracer import UnitfulTracer
from quantax.unitful.unitful import Unitful

AnyUnitType = Union[
    AnyArrayLike,
    Unitful,
    UnitfulTracer
]

def get_static_operand(
    x: UnitfulTracer | AnyArrayLike | Unitful,
) -> Unitful | None:
    if STATIC_OPTIM_STOP_FLAG:
        return None
    
    if isinstance(x, Unitful):
        if is_traced(x.val):
            return None
        if isinstance(x.val, jax.Array):
            if x.val.size >= MAX_STATIC_OPTIMIZED_SIZE:
                return None
            return Unitful(val=np.asarray(x.val, copy=True))
        return x

    # Physical arraylike without a unit
    if isinstance(x, AnyArrayLike):
        if is_traced(x):
            return None
        if isinstance(x, jax.Array):
            if x.size >= MAX_STATIC_OPTIMIZED_SIZE:
                return None
            return Unitful(val=np.asarray(x, copy=True))
        assert isinstance(x, StaticArrayLike), "Internal error, please report"
        return Unitful(val=x)

    # tracer
    assert isinstance(x, UnitfulTracer), "Internal error, please report"
    return x.static_unitful

    # Unitful
    x_arr = None
    if not is_traced(x.val):
        x_arr = x.val
        if isinstance(x_arr, jax.Array):
            if x_arr.size >= MAX_STATIC_OPTIMIZED_SIZE:
                return None
            x_arr = np.asarray(x_arr, copy=True)
    assert x_arr is None or isinstance(x_arr, StaticArrayLike)
    return x_arr
