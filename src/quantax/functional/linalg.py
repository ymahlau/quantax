# ruff: noqa: F811
from quantax.functional.utils import get_static_operand

import jax
import jax.numpy as jnp
import numpy as np
from plum import dispatch, overload

from quantax.core.typing import PhysicalArrayLike
from quantax.core.utils import is_traced
from quantax.unitful.unitful import Unitful


## norm #####################################
@overload
def norm(
    x: Unitful,
    *args,
    **kwargs,
) -> Unitful:
    new_val = jnp.linalg._orig_norm(x.val, *args, **kwargs)  # type: ignore
    # static computation
    new_static_arr = None
    if is_traced(new_val):
        x_arr = get_static_operand(x)
        if x_arr is not None:
            new_static_arr = np.linalg.norm(x_arr, *args, **kwargs)
    return Unitful(val=new_val, unit=x.unit, static_arr=new_static_arr)


@overload
def norm(
    x: PhysicalArrayLike,
    *args,
    **kwargs,
) -> jax.Array:
    return jnp.linalg._orig_norm(x, *args, **kwargs)  # type: ignore


@dispatch
def norm(
    x,
    *args,
    **kwargs,
):
    del x, args, kwargs
    raise NotImplementedError()
