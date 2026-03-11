from typing import Callable, get_args

from quantax.core.typing import AnyArrayLike, StaticArrayLike
from quantax.tracing.tracer import UnitfulTracer
from quantax.tracing.types import AnyUnitType
from quantax.tracing.utils import convert_input, get_static_operand
from quantax.unitful.unitful import Unitful


def binary_op_from_func(
    unitful_handler: Callable[[Unitful, Unitful], Unitful],
    tracer_handler: Callable[[UnitfulTracer, UnitfulTracer], UnitfulTracer],
    standard_handler: Callable[[AnyArrayLike, AnyArrayLike], AnyArrayLike],
    x: AnyUnitType,
    y: AnyUnitType,
) -> AnyUnitType:
    # input conversion to tracer if necessary
    x = convert_input(x)
    y = convert_input(y)
    
    # handle unitful inputs
    if isinstance(x, Unitful) and isinstance(y, Unitful):
        return unitful_handler(x, y)
    elif isinstance(x, Unitful) and not isinstance(y, (Unitful | UnitfulTracer)):
        y_unitful = Unitful(val=y)
        return unitful_handler(x, y_unitful)
    elif not isinstance(x, (Unitful | UnitfulTracer)) and isinstance(y, Unitful):
        x_unitful = Unitful(val=x)
        return unitful_handler(x_unitful, y)

    # handle tracer inputs
    if isinstance(x, UnitfulTracer) and isinstance(y, UnitfulTracer):
        return tracer_handler(x, y)
    if isinstance(x, UnitfulTracer):
        assert isinstance(y, StaticArrayLike | Unitful)
        y_tracer = UnitfulTracer(
            unit=None if not isinstance(y, Unitful) else y.unit, 
            static_unitful=get_static_operand(y), 
            value=y,
        )
        return tracer_handler(x, y_tracer)
    if isinstance(y, UnitfulTracer):
        assert isinstance(x, StaticArrayLike | Unitful)
        x_tracer = UnitfulTracer(
            unit=None if not isinstance(x, Unitful) else x.unit, 
            static_unitful=get_static_operand(x), 
            value=x,
        )
        return tracer_handler(x_tracer, y)

    # any other array-like
    assert isinstance(x, get_args(AnyArrayLike)), f"Invalid input type for multiply: {x}"
    assert isinstance(y, get_args(AnyArrayLike)), f"Invalid input type for multiply: {y}"
    result = standard_handler(x, y)
    assert isinstance(result, get_args(AnyArrayLike))
    return result



