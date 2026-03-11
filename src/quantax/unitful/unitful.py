from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self, get_args

import jax
import jax.numpy as jnp
import numpy as np
from pytreeclass import tree_repr

from quantax.core.pytrees import TreeClass, autoinit, frozen_field
from quantax.core.typing import (
    PHYSICAL_DTYPES,
    AnyArrayLike,
    NonPhysicalArrayLike,
    PhysicalArrayLike,
)
from quantax.core.unit import EMPTY_UNIT, Unit
from quantax.core.utils import (
    best_scale,
    is_traced,
)

if TYPE_CHECKING:
    from quantax.unitful.indexing import UnitfulIndexer


@autoinit
class Unitful(TreeClass):
    val: AnyArrayLike
    unit: Unit = frozen_field(default=EMPTY_UNIT)
    scale: int = frozen_field(default=0)
    optimize_scale: bool = frozen_field(default=True)

    # Tell NumPy that this class takes priority in operations
    __array_priority__ = 1000

    def _validate(self):
        bad_dtype = isinstance(self.val, (jax.Array, np.ndarray, np.number)) and self.dtype not in PHYSICAL_DTYPES
        if isinstance(self.val, get_args(NonPhysicalArrayLike)) or bad_dtype:
            if self.scale != 0:
                if isinstance(self.val, int):
                    raise Exception(f"Cannot have non-zero scale for integer {self}. Consider using float as input.")
                else:
                    raise Exception(f"Cannot have non-zero scale for non-physical {self}")
            if self.unit:
                raise Exception(f"Cannot have non-empty dimension for non-physical {self}")

    def __post_init__(self):
        from quantax.tracing.glob import get_global_replay_data

        self._validate()
        # we do not want to optimize the scale during replay. During replay we want to use the calculated values from MILP
        if get_global_replay_data() is not None:
            self.optimize_scale = False
        if not self.optimize_scale or not can_optimize_scale(self):
            return
        if not is_traced(self.val):
            # non-traced case: optimize scale
            assert isinstance(self.val, get_args(PhysicalArrayLike))
            optimized_val, power = best_scale(self.val, self.scale)
            self.val = optimized_val
            self.scale = self.scale - power

    def add_scale_offset(self, offset: int) -> Unitful:
        factor = 10 ** (-offset)
        return Unitful(
            val=self.val * factor,
            unit=self.unit,
            scale=self.scale + offset,
            optimize_scale=False,
        )

    def set_fixed_scale(self, new_scale: int) -> Unitful:
        if new_scale == self.scale:
            return self
        offset = new_scale - self.scale
        return self.add_scale_offset(offset)

    def materialise(self) -> AnyArrayLike:
        if self.unit:
            raise ValueError(f"Cannot materialise unitful array with a non-zero unit: {self.unit}")
        return self.value()

    def float_materialise(self) -> float:
        v = self.materialise()
        assert isinstance(v, float), f"safe float_materialise called on Unitful with non-float value: {self}"
        return v

    def array_materialise(self) -> jax.Array:
        v = self.materialise()
        assert isinstance(v, jax.Array), f"safe array_materialise called on Unitful with non-array value: {self}"
        return v

    def value(self) -> AnyArrayLike:
        if self.scale == 0:
            return self.val
        return self.val * (10**self.scale)

    def array_value(self) -> jax.Array:
        v = self.value()
        assert isinstance(v, jax.Array), f"safe array_value called on Unitful with non-array value: {self}"
        return v

    def float_value(self) -> float:
        v = self.value()
        assert isinstance(v, float), f"safe float_value called on Unitful with non-float value: {self}"
        return v

    @property
    def at(self) -> UnitfulIndexer:
        """Gets the indexer for this tree.

        Returns:
            UnitfulIndexer: Indexer that preserves type information
        """
        from quantax.unitful.indexing import UnitfulIndexer

        return UnitfulIndexer(self)

    @property
    def shape(self) -> tuple[int, ...]:
        if isinstance(self.val, (int, float, complex, bool)):
            return ()
        return self.val.shape

    @property
    def dtype(self):
        if not hasattr(self.val, "dtype"):
            raise Exception("Python scalar does not have dtype attribute")
        return self.val.dtype

    @property
    def ndim(self):
        if isinstance(self.val, (int, float, complex, bool)):
            return 0
        return self.val.ndim

    @property
    def size(self):
        if isinstance(self.val, (int, float, complex, bool)):
            return 1
        return self.val.size

    @property
    def T(self):
        raise NotImplementedError("TODO: call transpose here")

    def aset(
        self,
        attr_name: str,
        val: Any,
        create_new_ok: bool = False,
    ) -> Self:
        del attr_name, val, create_new_ok
        raise Exception(
            "the aset-method is unsafe for Unitful internals and therefore not implemented. "
            "Please use .at[].set() intead. Note that using aset on structures containing unitful is safe and "
            " implemented, just the internals of Unitful should not changed this way."
        )

    def __str__(self) -> str:
        try:
            return f"Unitful [{self.unit}]: {tree_repr(self.val)}"
        except Exception:
            return f"Unitful [{self.unit}]: {self.shape}"

    def __bool__(self) -> bool:
        if isinstance(self.val, (bool, np.bool_, jnp.bool_)):
            return bool(self.val)
        if isinstance(self.val, (np.ndarray, jnp.ndarray)):
            if self.val.size != 1:
                raise ValueError("Truth value of an array with multiple elements is ambiguous")
            return bool(self.val.item())
        return bool(self.val)

    def __repr__(self) -> str:
        return str(self)

    def __mul__(self, other: PhysicalArrayLike | "Unitful") -> "Unitful":
        from quantax.functional.numpy.basic import multiply

        return multiply(self, other)

    def __rmul__(self, other: PhysicalArrayLike | "Unitful") -> "Unitful":
        from quantax.functional.numpy.basic import multiply

        return multiply(other, self)

    def __truediv__(self, other: PhysicalArrayLike | "Unitful") -> "Unitful":
        from quantax.functional.numpy.basic import divide

        return divide(self, other)

    def __rtruediv__(self, other: PhysicalArrayLike | "Unitful") -> "Unitful":
        from quantax.functional.numpy.basic import divide

        return divide(other, self)

    def __add__(self, other: "Unitful | PhysicalArrayLike") -> "Unitful":
        from quantax.functional.numpy.basic import add

        return add(self, other)

    def __radd__(self, other: "Unitful | PhysicalArrayLike") -> "Unitful":
        from quantax.functional.numpy.basic import add

        return add(self, other)

    def __sub__(self, other: "Unitful | PhysicalArrayLike") -> "Unitful":
        from quantax.functional.numpy.basic import subtract

        return subtract(self, other)

    def __rsub__(self, other: "Unitful | PhysicalArrayLike") -> "Unitful":
        from quantax.functional.numpy.basic import subtract

        return subtract(other, self)

    def __eq__(self, other):
        from quantax.functional.numpy.comparisons import eq

        return eq(self, other)

    __hash__ = object.__hash__

    def __ne__(self, other):
        from quantax.functional.numpy.comparisons import ne

        return ne(self, other)

    def __lt__(self, other):
        from quantax.functional.numpy.comparisons import lt

        return lt(self, other)

    def __le__(self, other):
        from quantax.functional.numpy.comparisons import le

        return le(self, other)

    def __gt__(self, other):
        from quantax.functional.numpy.comparisons import gt

        return gt(self, other)

    def __ge__(self, other):
        from quantax.functional.numpy.comparisons import ge

        return ge(self, other)

    def __len__(self):
        if not isinstance(self.val, (jax.Array, np.ndarray)):
            return 1
        return len(self.val)

    def __getitem__(self, key: Any) -> "Unitful":
        """Enable numpy-style indexing"""
        if not isinstance(self.val, (jax.Array, np.ndarray)):
            raise Exception(f"Cannot slice Unitful with python scalar value ({self.val})")
        if isinstance(key, Unitful):
            key = key.materialise()
        if isinstance(key, tuple):
            new_list = []
            for k in key:
                if isinstance(k, Unitful):
                    new_list.append(k.materialise())
                else:
                    new_list.append(k)
            key = tuple(new_list)
        new_val = self.val[key]  # ty:ignore[invalid-argument-type]
        return Unitful(val=new_val, unit=self.unit)

    def __iter__(self):
        """Use a generator for simplicity"""
        if not isinstance(self.val, (jax.Array, np.ndarray)):
            raise Exception(f"Cannot iterate over Unitful with python scalar value ({self.val})")
        for v in self.val:
            yield (Unitful(val=v, unit=self.unit))

    def __reversed__(self):
        return iter(self[::-1])

    def __neg__(self) -> Unitful:
        if isinstance(self.val, get_args(NonPhysicalArrayLike)):
            raise Exception(f"Cannot perform negation on non-physcal value {self}")
        return Unitful(val=-self.val, unit=self.unit)

    def __pos__(self) -> Unitful:
        """Unary plus: +x"""
        if isinstance(self.val, get_args(NonPhysicalArrayLike)):
            raise Exception(f"Cannot perform unary plus on non-physcal value {self}")
        return Unitful(val=+self.val, unit=self.unit)


def can_optimize_scale(obj: Unitful | AnyArrayLike) -> bool:
    v = obj.val if isinstance(obj, Unitful) else obj
    if isinstance(v, get_args(NonPhysicalArrayLike)):
        return False
    if isinstance(v, (jax.Array, np.ndarray)) and v.dtype not in PHYSICAL_DTYPES:
        return False
    if is_traced(v):
        return False
    return True
