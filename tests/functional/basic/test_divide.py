import math

import jax
import jax.numpy as jnp
import numpy as np

from quantax.core.typing import SI
from quantax.core.unit import Unit
from quantax.functional import jit
from quantax.functional.numpy.basic import divide
from quantax.unitful.unitful import Unitful
from quantax.units import Hz, m, ms, s


def test_divide_unitful_same_unit():
    """Test division of two Unitful objects with same unit -> dimensionless"""
    a = 6 * s
    b = 2 * s

    result = divide(a, b)

    assert math.isclose(result.value(), 3.0)  # type: ignore
    assert result.unit == {}  # s^1 / s^1 = dimensionless


def test_divide_unitful_different_units():
    """Test division of two Unitful objects with different units"""
    dist = 10 * m
    time_val = 2 * s

    result = divide(dist, time_val)

    assert math.isclose(result.value(), 5.0)  # type: ignore
    assert result.unit == {SI.m: 1, SI.s: -1}  # m/s


def test_divide_unitful_different_scales():
    """Test division of Unitful objects with different scales"""
    a = 6 * s  # scale 0
    b = 2 * ms  # scale -3

    result = divide(a, b)

    assert math.isclose(result.value(), 3.0e3)  # type: ignore
    assert result.unit == {}  # dimensionless


def test_divide_unitful_unitful_python_python_type():
    """Test division with Python scalar values"""
    a = 6 * s
    b = 2 * s
    res = divide(a, b)

    assert math.isclose(res.value(), 3.0)  # type: ignore
    assert res.unit == {}


def test_divide_unitful_unitful_numpy_numpy_type():
    """Test division of Unitful objects with numpy array values"""
    a = s * np.array([6.0, 8.0, 10.0])
    b = s * np.array([2.0, 4.0, 5.0])
    res = divide(a, b)

    assert isinstance(res.val, np.ndarray)
    assert jnp.allclose(res.value(), jnp.array([3.0, 2.0, 2.0]))
    assert res.unit == {}


def test_divide_unitful_unitful_jax_jax_type():
    """Test division of Unitful objects with JAX array values"""
    a = s * jnp.array([6.0, 8.0, 10.0])
    b = s * jnp.array([2.0, 4.0, 5.0])
    res = divide(a, b)

    assert isinstance(res.val, jax.Array)
    assert jnp.allclose(res.value(), jnp.array([3.0, 2.0, 2.0]))
    assert res.unit == {}


def test_divide_unitful_unitful_mixed_jax_numpy_backend_priority():
    """Test that jax backend wins over numpy"""
    a = s * jnp.array([6.0, 8.0])
    b = s * np.array([2.0, 4.0])
    res = divide(a, b)
    assert isinstance(res.val, jax.Array)
    assert jnp.allclose(res.value(), jnp.array([3.0, 2.0]))
    assert res.unit == {}


def test_divide_arraylike_unitful():
    """Test division of plain scalar by Unitful (produces inverse unit)"""
    scalar = 6.0
    time_val = 2 * s

    result = divide(scalar, time_val)

    assert math.isclose(result.value(), 3.0)  # type: ignore
    assert result.unit == {SI.s: -1}


def test_divide_unitful_by_scalar():
    """Test division of Unitful by a plain scalar"""
    time_val = 6 * s
    scalar = 2.0

    result = divide(time_val, scalar)

    assert math.isclose(result.value(), 3.0)  # type: ignore
    assert result.unit == {SI.s: 1}


def test_divide_plain_jax_arrays():
    """Test plain jax / jax fallback"""
    x = jnp.array([6.0, 8.0, 10.0])
    y = jnp.array([2.0, 4.0, 5.0])

    result = divide(x, y)

    assert isinstance(result, jax.Array)
    assert jnp.allclose(result, jnp.array([3.0, 2.0, 2.0]))


def test_divide_plain_numpy_arrays():
    """Test plain numpy / numpy fallback"""
    x = np.array([6.0, 8.0, 10.0])
    y = np.array([2.0, 4.0, 5.0])

    result = divide(x, y)

    assert isinstance(result, np.ndarray)
    assert np.allclose(result, np.array([3.0, 2.0, 2.0]))


def test_divide_plain_python_scalars():
    """Test plain python scalar / scalar fallback"""
    assert divide(6, 2) == 3.0
    assert divide(7.0, 2.0) == 3.5


def test_divide_unit_inversion():
    """Test that dividing Hz by 1/s gives dimensionless"""
    # Hz is s^-1, so Hz / Hz should be dimensionless
    a = 10 * Hz
    b = 5 * Hz

    result = divide(a, b)

    assert math.isclose(result.value(), 2.0)  # type: ignore
    assert result.unit == {}


def test_divide_jitted():
    """Test jitting of division"""
    arr1 = s * jnp.array([6.0, 8.0, 10.0])
    arr2 = s * jnp.array([2.0, 4.0, 5.0])
    arr3 = s * jnp.array([1.0, 2.0, 2.0])

    def fn(a: Unitful, b: Unitful) -> Unitful:
        return divide(a, b)

    jitted_fn = jax.jit(fn)
    res1 = jitted_fn(arr1, arr2)
    res2 = jitted_fn(arr2, arr3)

    assert jnp.allclose(res1.value(), jnp.array([3.0, 2.0, 2.0]))
    assert jnp.allclose(res2.value(), jnp.array([2.0, 2.0, 2.5]))


_w_dimensionless = Unitful(val=2.0)


def _div_fn(x: Unitful, y: Unitful):
    def _inner(a):
        return divide(a, _w_dimensionless)  # _w captured as closure

    mid = divide(x, y)
    out = jit(_inner)(mid)
    return out


def test_complicated_divide_fn():
    x = Unitful(val=jnp.array([6.0, 8.0, 10.0]), unit=Unit({SI.s: 1}), scale=0)
    y = Unitful(val=jnp.array([2.0, 4.0, 5.0]), unit=Unit({SI.s: 1}), scale=0)
    res = jit(_div_fn)(x, y)
    res_no_jit = _div_fn(x, y)
    scale_diff = res.scale - res_no_jit.scale
    assert jnp.allclose(res.val, res_no_jit.val * (10**scale_diff))
