import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from quantax.core.typing import SI
from quantax.core.unit import Unit
from quantax.functional import jit
from quantax.functional.numpy.basic import add
from quantax.unitful.unitful import Unitful
from quantax.units import Hz, ms, s


def test_add_unitful_same_unit_same_scale():
    """Test addition of two Unitful objects with same unit and same scale"""
    a = 2 * s
    b = 3 * s

    result = add(a, b)

    assert math.isclose(result.value(), 5.0)  # type: ignore
    assert result.unit == {SI.s: 1}


def test_add_unitful_same_unit_different_scales():
    """Test addition of two Unitful objects with same unit but different scales"""
    a = 1 * s  # scale 0
    b = 1 * ms  # scale -3

    result = add(a, b)

    assert math.isclose(result.value(), 1.001)  # type: ignore
    assert result.unit == {SI.s: 1}


def test_add_unitful_different_units_raises():
    """Test that addition of Unitful arrays with different units raises ValueError"""
    time_val = 5 * s
    freq_val = 10 * Hz

    with pytest.raises(ValueError):
        add(time_val, freq_val)


def test_add_unitful_unitful_python_python_type():
    """Test addition of Unitful objects with Python scalar values"""
    a = 2 * s
    b = 3.5 * s
    res = add(a, b)

    assert math.isclose(res.value(), 5.5)  # type: ignore
    assert res.unit == {SI.s: 1}


def test_add_unitful_unitful_numpy_numpy_type():
    """Test addition of Unitful objects with numpy array values"""
    a = s * np.array([1.0, 2.0, 3.0])
    b = s * np.array([4.0, 5.0, 6.0])
    res = add(a, b)

    assert isinstance(res.val, np.ndarray)
    assert jnp.allclose(res.value(), jnp.array([5.0, 7.0, 9.0]))
    assert res.unit == {SI.s: 1}


def test_add_unitful_unitful_jax_jax_type():
    """Test addition of Unitful objects with JAX array values"""
    a = s * jnp.array([1.0, 2.0, 3.0])
    b = s * jnp.array([4.0, 5.0, 6.0])
    res = add(a, b)

    assert isinstance(res.val, jax.Array)
    assert jnp.allclose(res.value(), jnp.array([5.0, 7.0, 9.0]))
    assert res.unit == {SI.s: 1}


def test_add_unitful_unitful_mixed_jax_numpy_backend_priority():
    """Test that jax backend wins over numpy"""
    a = s * jnp.array([1.0, 2.0])
    b = s * np.array([3.0, 4.0])
    res1 = add(a, b)
    res2 = add(b, a)
    assert isinstance(res1.val, jax.Array)
    assert isinstance(res2.val, jax.Array)
    assert jnp.allclose(res1.value(), jnp.array([4.0, 6.0]))
    assert jnp.allclose(res2.value(), jnp.array([4.0, 6.0]))
    assert res1.unit == {SI.s: 1}


def test_add_arraylike_unitful():
    """Test addition of plain scalar with Unitful (treated as dimensionless)"""
    scalar = 5.0
    dimensionless = Unitful(val=3.0)

    result = add(scalar, dimensionless)

    assert math.isclose(result.value(), 8.0)  # type: ignore
    assert result.unit == {}


def test_add_plain_jax_arrays():
    """Test plain jax + jax fallback"""
    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.array([4.0, 5.0, 6.0])

    result = add(x, y)

    assert isinstance(result, jax.Array)
    assert jnp.allclose(result, jnp.array([5.0, 7.0, 9.0]))


def test_add_plain_numpy_arrays():
    """Test plain numpy + numpy fallback"""
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([4.0, 5.0, 6.0])

    result = add(x, y)

    assert isinstance(result, np.ndarray)
    assert np.allclose(result, np.array([5.0, 7.0, 9.0]))


def test_add_plain_python_scalars():
    """Test plain python scalar + scalar fallback"""
    assert add(2, 3) == 5
    assert add(2.0, 3.0) == 5.0
    assert add(1 + 2j, 3 + 4j) == (4 + 6j)


def test_add_jitted():
    """Test jitting of addition"""
    arr1 = s * jnp.array([1.0, 2.0, 3.0])
    arr2 = s * jnp.array([4.0, 5.0, 6.0])
    arr3 = s * jnp.array([1.0, 1.0, 1.0])

    def fn(a: Unitful, b: Unitful) -> Unitful:
        return add(a, b)

    jitted_fn = jax.jit(fn)
    res1 = jitted_fn(arr1, arr2)
    res2 = jitted_fn(arr2, arr3)

    assert jnp.allclose(res1.value(), jnp.array([5.0, 7.0, 9.0]))
    assert jnp.allclose(res2.value(), jnp.array([5.0, 6.0, 7.0]))


_w = Unitful(val=3.0, unit=Unit({SI.s: 1}), scale=0)


def _add_fn(x: Unitful, y: Unitful):
    def _inner(a):
        return add(a, _w)  # _w captured as closure

    mid = add(x, y)
    out = jit(_inner)(mid)
    return out


def test_complicated_add_fn():
    x = Unitful(val=jnp.array([1.0, 2.0, 3.0]), unit=Unit({SI.s: 1}), scale=0)
    y = Unitful(val=jnp.array([4.0, 5.0, 6.0]), unit=Unit({SI.s: 1}), scale=0)
    res = jit(_add_fn)(x, y)
    res_no_jit = _add_fn(x, y)
    scale_diff = res.scale - res_no_jit.scale
    assert jnp.allclose(res.val, res_no_jit.val * (10**scale_diff))
