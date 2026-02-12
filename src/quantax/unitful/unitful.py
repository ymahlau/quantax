from __future__ import annotations
from quantax.core.unit import EMPTY_UNIT

from typing import TYPE_CHECKING, Any, Self

import jax
import jax.numpy as jnp
import numpy as np
from pytreeclass import tree_repr

from quantax.core.fraction import IntFraction
from quantax.core.pytrees import TreeClass, autoinit, frozen_field
from quantax.core.typing import (
    PHYSICAL_DTYPES,
    NonPhysicalArrayLike,
    PhysicalArrayLike,
    RealPhysicalArrayLike,
    StaticArrayLike,
    StaticPhysicalArrayLike, 
    AnyArrayLike,
)
from quantax.core.utils import (
    best_scale,
    is_currently_compiling,
    is_traced,
)
from quantax.core.unit import Unit

if TYPE_CHECKING:
    from quantax.unitful.indexing import UnitfulIndexer


@autoinit
class Unitful(TreeClass):
    val: AnyArrayLike
    unit: Unit = frozen_field(default=EMPTY_UNIT)
    scale: int = frozen_field(default=0)
    optimize_scale: bool = frozen_field(default=True)

    def _validate(self):
        bad_dtype = isinstance(self.val, jax.Array | np.ndarray | np.number) and self.dtype not in PHYSICAL_DTYPES
        if isinstance(self.val, NonPhysicalArrayLike) or bad_dtype:
            if self.scale != 0:
                if isinstance(self.val, int):
                    raise Exception(f"Cannot have non-zero scale for integer {self}. Consider using float as input.")
                else:
                    raise Exception(f"Cannot have non-zero scale for non-physical {self}")
            if self.unit:
                raise Exception(f"Cannot have non-empty dimension for non-physical {self}")
                

    def __post_init__(self):
        self._validate()
        if not self.optimize_scale or not can_optimize_scale(self):
            return
        if not is_traced(self.val):
            # non-traced case: optimize scale
            assert isinstance(self.val, PhysicalArrayLike)
            optimized_val, power = best_scale(self.val, self.scale)
            self.val = optimized_val
            self.scale = self.scale - power

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
        if isinstance(self.scale, IntFraction):
            scale = self.scale.value()
        if scale == 0:
            return self.val
        return self.val * (10**scale)

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
        if isinstance(self.val, int | float | complex | bool):
            return ()
        return self.val.shape

    @property
    def dtype(self):
        if isinstance(self.val, int | float | complex | bool):
            raise Exception("Python scalar does not have dtype attribute")
        return self.val.dtype

    @property
    def ndim(self):
        if isinstance(self.val, int | float | complex | bool):
            return 0
        return self.val.ndim

    @property
    def size(self):
        if isinstance(self.val, int | float | complex | bool):
            return 1
        return self.val.size

    @property
    def T(self):
        raise NotImplementedError("TODO: call transpose here")
        # if not isinstance(self.val, jax.Array | np.ndarray | np.number | np.bool):
        #     raise Exception(f"Cannot call .T on {self}")
        # new_val = self.val.T
        # new_static_arr = None
        # if is_traced(new_val):
        #     arr = get_static_operand(self)
        #     if arr is not None:
        #         assert isinstance(arr, np.ndarray | np.number | np.bool)
        #         new_static_arr = arr.T
        # return Unitful(val=new_val, unit=self.unit, static_arr=new_static_arr)

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

    def astype(self, *args, **kwargs) -> "Unitful":
        from quantax.functional.numpy import astype

        return astype(self, *args, **kwargs)

    def squeeze(
        self,
        axis: int | None = None,
    ) -> "Unitful":
        from quantax.functional.numpy import squeeze

        return squeeze(self, axis)

    def reshape(
        self,
        *args,
        **kwargs,
    ) -> "Unitful":
        from quantax.functional.numpy import reshape

        return reshape(self, args, **kwargs)

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
        from quantax.functional.numpy import multiply

        return multiply(self, other)

    def __rmul__(self, other: PhysicalArrayLike | "Unitful") -> "Unitful":
        from quantax.functional.numpy import multiply

        return multiply(other, self)

    def __truediv__(self, other: PhysicalArrayLike | "Unitful") -> "Unitful":
        from quantax.functional.numpy import divide

        return divide(self, other)

    def __rtruediv__(self, other: PhysicalArrayLike | "Unitful") -> "Unitful":
        from quantax.functional.numpy import divide

        return divide(other, self)

    def __add__(self, other: "Unitful | PhysicalArrayLike") -> "Unitful":
        from quantax.functional.numpy import add

        return add(self, other)

    def __radd__(self, other: "Unitful | PhysicalArrayLike") -> "Unitful":
        from quantax.functional.numpy import add

        return add(self, other)

    def __sub__(self, other: "Unitful | PhysicalArrayLike") -> "Unitful":
        from quantax.functional.numpy import subtract

        return subtract(self, other)

    def __rsub__(self, other: "Unitful | PhysicalArrayLike") -> "Unitful":
        from quantax.functional.numpy import subtract

        return subtract(other, self)

    def __lt__(self, other: "Unitful | RealPhysicalArrayLike") -> "Unitful":
        from quantax.functional.numpy import lt

        return lt(self, other)

    def __le__(self, other: "Unitful | RealPhysicalArrayLike") -> "Unitful":
        from quantax.functional.numpy import le

        return le(self, other)

    def __eq__(self, other: "Unitful | RealPhysicalArrayLike") -> "Unitful":  # type: ignore[override]
        from quantax.functional.numpy import eq

        return eq(self, other)

    def __ne__(self, other: "Unitful | RealPhysicalArrayLike") -> "Unitful":  # type: ignore[override]  # allow non-bool return for array-style comparison
        from quantax.functional.numpy import ne

        return ne(self, other)

    def __ge__(self, other: "Unitful | RealPhysicalArrayLike") -> "Unitful":
        from quantax.functional.numpy import ge

        return ge(self, other)

    def __gt__(self, other: "Unitful | RealPhysicalArrayLike") -> "Unitful":
        from quantax.functional.numpy import gt

        return gt(self, other)

    def __len__(self):
        if not isinstance(self.val, jax.Array | np.ndarray):
            return 1
        return len(self.val)

    def __getitem__(self, key: Any) -> "Unitful":
        """Enable numpy-style indexing"""
        if not isinstance(self.val, jax.Array | np.ndarray):
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
        if not isinstance(self.val, jax.Array | np.ndarray):
            raise Exception(f"Cannot iterate over Unitful with python scalar value ({self.val})")
        for v in self.val:
            yield (Unitful(val=v, unit=self.unit))

    def __reversed__(self):
        return iter(self[::-1])

    def __neg__(self) -> Unitful:
        if isinstance(self.val, NonPhysicalArrayLike):
            raise Exception(f"Cannot perform negation on non-physcal value {self}")
        return Unitful(val=-self.val, unit=self.unit)  # ty:ignore[unsupported-operator]

    def __pos__(self) -> Unitful:
        """Unary plus: +x"""
        if isinstance(self.val, NonPhysicalArrayLike):
            raise Exception(f"Cannot perform unary plus on non-physcal value {self}")
        return Unitful(val=+self.val, unit=self.unit)  # ty:ignore[unsupported-operator]

    def __abs__(self):
        from quantax.functional.numpy import abs_impl

        return abs_impl(self)

    def __matmul__(self, other: "Unitful") -> "Unitful":
        from quantax.functional.numpy import matmul

        return matmul(self, other)

    def __pow__(self, other: int) -> "Unitful":
        from quantax.functional.numpy import pow

        return pow(self, other)

    def min(self, **kwargs) -> "Unitful":
        from quantax.functional.numpy import min

        return min(self, **kwargs)

    def max(self, **kwargs) -> "Unitful":
        from quantax.functional.numpy import max

        return max(self, **kwargs)

    def mean(self, **kwargs) -> "Unitful":
        from quantax.functional.numpy import mean

        return mean(self, **kwargs)

    def sum(self, **kwargs) -> "Unitful":
        from quantax.functional.numpy import sum

        return sum(self, **kwargs)

    def prod(self, **kwargs) -> "Unitful":
        from quantax.functional.numpy import prod

        return prod(self, **kwargs)

    def argmax(self, **kwargs) -> "Unitful":
        from quantax.functional.numpy import argmax

        return argmax(self, **kwargs)

    def argmin(self, **kwargs) -> "Unitful":
        from quantax.functional.numpy import argmin

        return argmin(self, **kwargs)


# This conversion method is necessary, because within jit-context we lie to the dispatcher.
# Specifically, functions that are supposed to return a jax array will return a unitful to be able to perform
# scale optimization.
# def unitful_to_array_conversion(obj: Unitful):
#     assert obj.unit == {}
#     if is_currently_compiling():
#         return obj
#     return obj.array_materialise()


# add_conversion_method(type_from=Unitful, type_to=jax.Array, f=unitful_to_array_conversion)


# def unitful_to_array_conversion_with_bool(obj: Unitful) -> np.bool | np.ndarray | jax.Array | bool:
#     assert obj.unit == {}
#     if is_currently_compiling():
#         result: np.bool | np.ndarray | jax.Array | bool = obj  # type: ignore
#     else:
#         result: np.bool | np.ndarray | jax.Array | bool = obj.materialise()  # type: ignore
#     return result


# add_conversion_method(
#     type_from=Unitful,
#     type_to=np.bool | np.ndarray | jax.Array | bool,  # type: ignore
#     f=unitful_to_array_conversion_with_bool,
# )


def can_optimize_scale(obj: Unitful | AnyArrayLike) -> bool:
    v = obj.val if isinstance(obj, Unitful) else obj
    if isinstance(v, NonPhysicalArrayLike):
        return False
    if isinstance(v, jax.Array | np.ndarray) and v.dtype not in PHYSICAL_DTYPES:
        return False
    if is_traced(v) and not isinstance(obj, Unitful):
        return False
    if is_traced(v):
        return False
    return True
