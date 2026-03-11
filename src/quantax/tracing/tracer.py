from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, get_args

import jax
import numpy as np

from quantax.core.typing import AnyArrayLike, StaticArrayLike
from quantax.core.unit import Unit
from quantax.unitful.unitful import Unitful


@dataclass(kw_only=True)
class UnitfulTracer:
    # if None, the value was not a unitful before and needs to be converted back after replay
    unit: Unit | None = field()
    val_shape_dtype: jax.ShapeDtypeStruct | None = None
    id: int = field(default=-1, init=False)
    static_unitful: Unitful | None = None  # used during tracing to compute optimal scales

    # actual value, if tracer holds constant value (e.g. closure value from outer scope)
    value: Unitful | AnyArrayLike | None = field(default=None)

    def __post_init__(self):
        # tracer registers itself. local import cannot be avoided here
        from quantax.tracing.glob import register_tracer
        register_tracer(self)
        
        if self.static_unitful is not None:
            # if the tracer is coming from a global jax value (closure), then we need to convert the jax value to numpy
            # this does not change the function behavior, because global jax values are treated as constants anyways
            if isinstance(self.static_unitful.val, jax.Array):
                new_val = np.asarray(self.static_unitful.val)
                self.static_unitful = self.static_unitful.updated_copy(val=new_val)

            assert isinstance(self.static_unitful.val, get_args(StaticArrayLike))

    def __mul__(self, other: Any) -> UnitfulTracer:
        from quantax.functional.numpy.basic import multiply

        return multiply(self, other)  # ty:ignore[no-matching-overload]

    def __rmul__(self, other: Any) -> UnitfulTracer:
        from quantax.functional.numpy.basic import multiply

        return multiply(other, self)  # ty:ignore[no-matching-overload]



