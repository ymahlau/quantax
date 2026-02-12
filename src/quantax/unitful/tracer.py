from __future__ import annotations
import numpy as np
from quantax.core.unit import Unit

from dataclasses import dataclass, field
from typing import Any

import jax

from quantax.core.glob import register_tracer
from quantax.core.typing import StaticArrayLike
from quantax.unitful.unitful import Unitful

@dataclass(kw_only=True)
class OperatorNode:
    op_name: str
    args: dict[str, Any]
    output_tracer: tuple[UnitfulTracer, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    id: int = field(default=-1, init=False)

    
@dataclass(kw_only=True)
class UnitfulTracer:
    unit: Unit = field()
    val_shape_dtype: jax.ShapeDtypeStruct | None = None
    static_unitful: Unitful | None = None
    parent: OperatorNode | None = None
    id: int = field(default=-1, init=False)

    def __post_init__(self):
        self.id = register_tracer(self)
        if self.static_unitful is not None:
            assert isinstance(self.static_unitful.val, np.ndarray | np.number)
            
    def __mul__(self, other: Any) -> UnitfulTracer:
        from quantax.functional.numpy import multiply
        
        return multiply(self, other)

    def __rmul__(self, other: Any) -> UnitfulTracer:
        from quantax.functional.numpy import multiply
        
        return multiply(other, self)
            
    
