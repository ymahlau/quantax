from quantax.core.unit import Unit
from quantax.functional import jit
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from quantax.core.typing import SI
from quantax.functional.numpy.basic import subtract
from quantax.unitful.unitful import Unitful
from quantax.units import Hz, ms, s, m


def test_subtract_unitful_same_unit_same_scale():
    """Test subtraction of two Unitful objects with same unit and same scale"""
    a = 5 * s
    b = 3 * s

    result = subtract(a, b)

    assert math.isclose(result.value(), 2.0)  # type: ignore
    assert result.unit == {SI.s: 1}


def test_subtract_unitful_same_unit_different_scales():
    """Test subtraction of two Unitful objects with same unit but different scales"""
    a = 1 * s     # scale 0, value = 1
    b = 1 * ms    # scale -3, value = 0.001

    result = subtract(a, b)

    assert math.isclose(result.value(), 0.999)  # type: ignore
    assert result.unit == {SI.s: 1}


def test_subtract_unitful_different_units_raises():
    """Test that subtraction of Unitful arrays with different units raises ValueError"""
    time_val = 5 * s
    freq_val = 10 * Hz

    with pytest.raises(ValueError):
        subtract(time_val, freq_val)


def test_subtract_unitful_unitful_python_python_type():
    """Test subtraction of Unitful objects with Python scalar values"""
    a = 5 * s
    b = 2.5 * s
    res = subtract(a, b)

    assert math.isclose(res.value(), 2.5)  # type: ignore
    assert res.unit == {SI.s: 1}


def test_subtract_unitful_unitful_numpy_numpy_type():
    """Test subtraction of Unitful objects with numpy array values"""
    a = s * np.array([5.0, 7.0, 9.0])
    b = s * np.array([1.0, 2.0, 3.0])
    res = subtract(a, b)

    assert isinstance(res.val, np.ndarray)
    assert jnp.allclose(res.value(), jnp.array([4.0, 5.0, 6.0]))
    assert res.unit == {SI.s: 1}


def test_subtract_unitful_unitful_jax_jax_type():
    """Test subtraction of Unitful objects with JAX array values"""
    a = s * jnp.array([5.0, 7.0, 9.0])
    b = s * jnp.array([1.0, 2.0, 3.0])
    res = subtract(a, b)

    assert isinstance(res.val, jax.Array)
    assert jnp.allclose(res.value(), jnp.array([4.0, 5.0, 6.0]))
    assert res.unit == {SI.s: 1}


def test_subtract_unitful_unitful_mixed_jax_numpy_backend_priority():
    """Test that jax backend wins over numpy"""
    a = s * jnp.array([5.0, 7.0])
    b = s * np.array([1.0, 2.0])
    res1 = subtract(a, b)
    assert isinstance(res1.val, jax.Array)
    assert jnp.allclose(res1.value(), jnp.array([4.0, 5.0]))
    assert res1.unit == {SI.s: 1}


def test_subtract_arraylike_unitful():
    """Test subtraction of plain scalar with dimensionless Unitful"""
    scalar = 5.0
    dimensionless = Unitful(val=3.0)

    result = subtract(scalar, dimensionless)

    assert math.isclose(result.value(), 2.0)  # type: ignore
    assert result.unit == {}


def test_subtract_plain_jax_arrays():
    """Test plain jax - jax fallback"""
    x = jnp.array([5.0, 7.0, 9.0])
    y = jnp.array([1.0, 2.0, 3.0])

    result = subtract(x, y)

    assert isinstance(result, jax.Array)
    assert jnp.allclose(result, jnp.array([4.0, 5.0, 6.0]))


def test_subtract_plain_numpy_arrays():
    """Test plain numpy - numpy fallback"""
    x = np.array([5.0, 7.0, 9.0])
    y = np.array([1.0, 2.0, 3.0])

    result = subtract(x, y)

    assert isinstance(result, np.ndarray)
    assert np.allclose(result, np.array([4.0, 5.0, 6.0]))


def test_subtract_plain_python_scalars():
    """Test plain python scalar - scalar fallback"""
    assert subtract(5, 3) == 2
    assert subtract(5.0, 2.5) == 2.5
    assert subtract(4 + 6j, 1 + 2j) == (3 + 4j)


def test_subtract_jitted():
    """Test jitting of subtraction"""
    arr1 = s * jnp.array([5.0, 7.0, 9.0])
    arr2 = s * jnp.array([1.0, 2.0, 3.0])
    arr3 = s * jnp.array([1.0, 1.0, 1.0])

    def fn(a: Unitful, b: Unitful) -> Unitful:
        return subtract(a, b)

    jitted_fn = jax.jit(fn)
    res1 = jitted_fn(arr1, arr2)
    res2 = jitted_fn(arr2, arr3)

    assert jnp.allclose(res1.value(), jnp.array([4.0, 5.0, 6.0]))
    assert jnp.allclose(res2.value(), jnp.array([0.0, 1.0, 2.0]))


_w = Unitful(val=1.0, unit=Unit({SI.s: 1}), scale=0)


def _sub_fn(x: Unitful, y: Unitful):
    def _inner(a):
        return subtract(a, _w)  # _w captured as closure
    mid = subtract(x, y)
    out = jit(_inner)(mid)
    return out


def test_complicated_subtract_fn():
    x = Unitful(val=jnp.array([5.0, 7.0, 9.0]), unit=Unit({SI.s: 1}), scale=0)
    y = Unitful(val=jnp.array([1.0, 2.0, 3.0]), unit=Unit({SI.s: 1}), scale=0)
    res = jit(_sub_fn)(x, y)
    res_no_jit = _sub_fn(x, y)
    scale_diff = res.scale - res_no_jit.scale
    assert jnp.allclose(res.val, res_no_jit.val * (10 ** scale_diff))
