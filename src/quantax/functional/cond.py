from quantax.unitful.utils import check_jax_unitful_tracer_type
from typing import Callable, TypeVar

import jax
import numpy as np
from typing_extensions import TypeVarTuple, Unpack

from quantax.unitful.unitful import Unitful

Ts = TypeVarTuple("Ts")
R = TypeVar("R")


def cond(
    pred: bool | jax.Array | np.bool_ | Unitful,
    true_fun: Callable[Unpack[Ts], R],
    false_fun: Callable[Unpack[Ts], R],
    *operands: Unpack[Ts],
) -> R:
    # test for jax/unitful components
    has_jax, has_unitful, has_tracer = check_jax_unitful_tracer_type(data=(pred, operands))

    # CASE 1: traced implementation. trace each function and return result.
    if has_tracer:
        pass

    # CASE 2: unitful eager implementation. start tracing mechanism, solve MILP and replay
    if has_unitful:
        pass

    # CASE 3: standard jax call, todo: use original function similar to how jit impl. does it
    return jax.lax.cond(
        pred,
        true_fun,
        false_fun,
        *operands,
    )
