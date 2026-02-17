from __future__ import annotations
from typing import Mapping
from frozendict import frozendict

from quantax.core.fraction import IntFraction
from quantax.core.pytrees import TreeClass, autoinit, frozen_field
from quantax.core.typing import SI


class Unit(frozendict[SI, int | IntFraction]):
    def __new__(
        cls, 
        mapping: Mapping[SI, int | IntFraction],
    ):
        # Run Validation Logic
        for k, v in mapping.items():
            if not isinstance(k, SI):
                raise TypeError(f"Key {k} must be SI")
            if not isinstance(v, (IntFraction, int)):
                raise TypeError(f"Value {v} must be Fraction or int")
        
        return super().__new__(cls, mapping)  # ty:ignore[no-matching-overload]
    
    def __str__(self) -> str:
        if not self.items():
            return "{}"
        res_str = "{"
        for k, v in self.items():
            res_str += f"{k.name}^{v} * "
        res_str = res_str[:-3]  # remove last three chars " * "
        res_str += "}"
        return res_str
    
    def __repr__(self) -> str:
        return str(self)


EMPTY_UNIT = Unit({})
