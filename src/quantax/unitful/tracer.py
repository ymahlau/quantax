from __future__ import annotations
from quantax.core.glob import register_tracer
from typing import Any
from dataclasses import dataclass, field
import jax
from jaxtyping import ArrayLike
from quantax.core.typing import StaticArrayLike
from quantax.core import glob
from quantax.core.pytrees import autoinit, frozen_field, frozen_private_field
from quantax.unitful.unitful import Unitful


@autoinit
class UnitfulTracer(Unitful):
    val_shape_dtype: jax.ShapeDtypeStruct | None = None
    static_arr: StaticArrayLike | None = None
    val: StaticArrayLike | None = None
    parent: OperatorNode | None = None
    id: int = frozen_private_field(default=-1)
    
    def __post_init__(self):
        self.id = register_tracer(self)
        a = 1
    

@dataclass(kw_only=True)
class OperatorNode:
    op_name: str
    args: dict[str, Any]
    output_tracer: tuple[UnitfulTracer, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


