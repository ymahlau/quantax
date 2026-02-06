from frozendict import frozendict

from quantax.core.fraction import Fraction
from quantax.core.pytrees import TreeClass, autoinit, frozen_field
from quantax.core.typing import SI


@autoinit
class Unit(TreeClass):
    scale: int = frozen_field()
    dim: dict[SI, int | Fraction] | frozendict[SI, int | Fraction] = frozen_field()

    def __post_init__(self):
        if isinstance(self.dim, dict) and not isinstance(self.dim, frozendict):
            self.dim = frozendict(self.dim)

    def __str__(self) -> str:
        res_str = f"10^{self.scale} " if self.scale != 0 else ""
        for k, v in self.dim.items():
            res_str += f"{k.name}^{v} " if v != 1 else f"{k.name} "
        return res_str[:-1]

    def __repr__(self) -> str:
        return str(self)

    def __hash__(self) -> int:
        return hash((hash(self.dim), self.scale))


EMPTY_UNIT = Unit(scale=0, dim={})
