from quantax.unitful.alignment import align_scales
from dataclasses import dataclass
from typing import Any

import jax
import numpy as np

from quantax.unitful.unitful import Unitful


@dataclass(frozen=True)
class UnitfulIndexer:
    unitful: Unitful
    where: Any | None = None

    def __getitem__(self, where: Any) -> "UnitfulIndexer":
        if self.where is not None:
            raise Exception("Already called [] on Unitful.at! Double Indexing [][] is currently not supported")
        return UnitfulIndexer(self.unitful, where)

    def get(self) -> Unitful:
        """Get the leaf values at the specified location."""
        if self.where is None:
            return self.unitful
        if not isinstance(self.unitful.val, jax.Array | np.ndarray):
            raise Exception(f"Cannot index scalar value: {self.unitful}")
        new_val = self.unitful.val[self.where]  # type: ignore
        return Unitful(val=new_val, unit=self.unitful.unit)

    def set(self, value: Unitful) -> Unitful:
        """Set the leaf values at the specified location."""
        if self.where is None:
            raise Exception("Cannot update value if no where clause is given")
        if not isinstance(self.unitful.val, jax.Array | np.ndarray):
            raise Exception(f"Cannot index scalar value: {self.unitful}")
        if self.unitful.unit.dim != value.unit.dim:
            raise Exception(
                f"Cannot update array value with different unit: {self.unitful.unit.dim} != {value.unit.dim}"
            )
        align_self, align_other = align_scales(self.unitful, value)
        align_self_arr = align_self.val
        assert isinstance(align_self_arr, jax.Array | np.ndarray)
        if isinstance(align_self_arr, np.ndarray):
            new_val = np.copy(align_self_arr)
            new_val[self.where] = align_other.val
        else:
            new_val = align_self_arr.at[self.where].set(align_other.val)
        return Unitful(val=new_val, unit=align_self.unit)

    def add(self, value: "Unitful") -> "Unitful":
        """Set the leaf values at the specified location."""
        if self.where is None:
            raise Exception("Cannot update value if no where clause is given")
        if not isinstance(self.unitful.val, jax.Array | np.ndarray):
            raise Exception(f"Cannot index scalar value: {self.unitful}")
        if self.unitful.unit.dim != value.unit.dim:
            raise Exception(
                f"Cannot update array value with different unit: {self.unitful.unit.dim} != {value.unit.dim}"
            )
        align_self, align_other = align_scales(self.unitful, value)
        align_self_arr = align_self.val
        assert isinstance(align_self_arr, jax.Array | np.ndarray)
        if isinstance(align_self_arr, np.ndarray):
            new_val = np.copy(align_self_arr)
            new_val[self.where] += align_other.val
        else:
            new_val = align_self_arr.at[self.where].add(align_other.val)
        return Unitful(val=new_val, unit=align_self.unit)

    def subtract(self, value: Unitful) -> Unitful:
        """Set the leaf values at the specified location."""
        if self.where is None:
            raise Exception("Cannot update value if no where clause is given")
        if not isinstance(self.unitful.val, jax.Array | np.ndarray):
            raise Exception(f"Cannot index scalar value: {self.unitful}")
        if self.unitful.unit.dim != value.unit.dim:
            raise Exception(
                f"Cannot update array value with different unit: {self.unitful.unit.dim} != {value.unit.dim}"
            )
        align_self, align_other = align_scales(self.unitful, value)
        align_self_arr = align_self.val
        assert isinstance(align_self_arr, jax.Array | np.ndarray)
        if isinstance(align_self_arr, np.ndarray):
            new_val = np.copy(align_self_arr)
            new_val[self.where] += align_other.val
        else:
            new_val = align_self_arr.at[self.where].subtract(align_other.val)
        return Unitful(val=new_val, unit=align_self.unit)

    def multiply(self, value: jax.Array) -> "Unitful":
        if isinstance(value, Unitful):
            raise Exception(
                "Multiplying part of an array with another Unitful would lead to different units within the Array!"
            )
        if self.where is None:
            raise Exception("Cannot update value if no where clause is given")
        if not isinstance(self.unitful.val, jax.Array | np.ndarray):
            raise Exception(f"Cannot index scalar value: {self.unitful}")
        v = self.unitful.val
        if isinstance(v, np.ndarray):
            new_val = np.copy(self.unitful.val)
            new_val[self.where] *= value
        else:
            new_val = v.at[self.where].multiply(value)
        return Unitful(val=new_val, unit=self.unitful.unit)

    def divide(self, value: jax.Array) -> Unitful:
        if isinstance(value, Unitful):
            raise Exception(
                "Multiplying part of an array with another Unitful would lead to different units within the Array!"
            )
        if self.where is None:
            raise Exception("Cannot update value if no where clause is given")
        if not isinstance(self.unitful.val, jax.Array | np.ndarray):
            raise Exception(f"Cannot index scalar value: {self.unitful}")
        v = self.unitful.val
        if isinstance(v, np.ndarray):
            new_val = np.copy(self.unitful.val)
            new_val[self.where] /= value
        else:
            new_val = v.at[self.where].divide(value)
        return Unitful(val=new_val, unit=self.unitful.unit)

    def power(self, value: Any) -> Unitful:
        del value
        raise Exception("Raising part of an array to a power is an undefined operation for a Unitful")
