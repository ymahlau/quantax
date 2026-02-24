from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import jax

from quantax.core.glob import register_tracer
from quantax.core.typing import AnyArrayLike, StaticArrayLike
from quantax.core.unit import Unit
from quantax.unitful.unitful import Unitful
import numpy as np



@dataclass(kw_only=True)
class UnitfulTracer:
    unit: Unit | None = field()
    val_shape_dtype: jax.ShapeDtypeStruct | None = None
    id: int = field(default=-1, init=False)
    static_unitful: Unitful | None = None  # used during tracing to compute optimal scales
    value: Unitful | AnyArrayLike | None = field(default=None)  # value is only used during replay, not before

    def __post_init__(self):
        register_tracer(self)
        if self.static_unitful is not None:
            # if the tracer is coming from a global jax value (closure), then we need to convert the jax value to numpy
            # this does not change the function behavior, because global jax values are treated as constants anyways
            if isinstance(self.static_unitful.val, jax.Array):
                new_val = np.asarray(self.static_unitful.val)
                self.static_unitful = self.static_unitful.updated_copy(val=new_val)

            assert isinstance(self.static_unitful.val, StaticArrayLike)

    def __mul__(self, other: Any) -> UnitfulTracer:
        from quantax.functional.numpy import multiply

        return multiply(self, other)  # ty:ignore[no-matching-overload]

    def __rmul__(self, other: Any) -> UnitfulTracer:
        from quantax.functional.numpy import multiply

        return multiply(other, self)  # ty:ignore[no-matching-overload]
