from typing import Union

from quantax.core.typing import AnyArrayLike
from quantax.tracing.tracer import UnitfulTracer
from quantax.unitful.unitful import Unitful

AnyUnitType = Union[AnyArrayLike, Unitful, UnitfulTracer]
