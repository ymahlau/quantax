import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from quantax.core.typing import SI
from quantax.core.unit import Unit
from quantax.functional import jit
from quantax.functional.numpy.basic import multiply
from quantax.unitful.unitful import Unitful
from quantax.units import Hz, ms, s


def test_multiply_unitful_unitful_same_dimensions():
    """Test multiplication of two Unitful objects with same dimensions"""
    time1 = s * 2
    time2 = 3 * s

    result = multiply(time1, time2)

    assert jnp.allclose(result.value(), 6.0)
    assert result.unit == {SI.s: 2}


def test_multiply_unitful_unitful_different_dimensions():
    """Test multiplication of two Unitful objects with different dimensions"""
    time_val = 5 * s
    freq_val = 10 * Hz

    result = multiply(time_val, freq_val)

    assert math.isclose(result.value(), 50.0)  # type: ignore
    assert result.unit == {}  # s^1 * s^-1 = dimensionless


def test_multiply_unitful_unitful_different_scales():
    """Test multiplication of Unitful objects with different scales"""
    time1 = 2 * s  # scale 0
    time2 = 3 * ms  # scale -3

    result = multiply(time1, time2)

    assert math.isclose(result.value(), 6.0e-3)  # type: ignore
    assert result.unit == {SI.s: 2}


def test_multiply_unitful_unitful_python_python_type_and_dim():
    """Test multiplication of Unitful objects with value Python scalar and Python scalar"""
    a = 2 * s
    b = 3.5 * s
    res = multiply(a, b)

    assert math.isclose(res.value(), 7.0)  # type: ignore
    assert res.unit == {SI.s: 2}
    assert isinstance(res.val, float)


def test_multiply_unitful_unitful_numpy_numpy_type_and_dim():
    """Test multiplication of Unitful objects with value numpy Array and numpy Array"""
    a = s * np.array([1.0, 2.0, 3.0])
    b = s * np.array([4.0, 5.0, 6.0])
    res = multiply(a, b)

    assert isinstance(res.val, np.ndarray)
    assert jnp.allclose(res.value(), jnp.array([4.0, 10.0, 18.0]))
    assert res.unit == {SI.s: 2}


def test_multiply_unitful_unitful_jax_jax_type_and_dim():
    """Test multiplication of Unitful objects with value Jax Array and Jax Array"""
    a = s * jnp.array([1.0, 2.0, 3.0])
    b = s * jnp.array([4.0, 5.0, 6.0])
    res = multiply(a, b)

    assert isinstance(res.val, jax.Array)
    assert jnp.allclose(res.value(), jnp.array([4.0, 10.0, 18.0]))
    assert res.unit == {SI.s: 2}


def test_multiply_with_arrays():
    """Test multiplication with array values"""
    time_array = s * jnp.array([1.0, 2.0, 3.0])
    freq_array: Unitful = jnp.array([2.0, 3.0, 4.0]) * Hz
    assert isinstance(freq_array, Unitful)
    assert isinstance(time_array, Unitful)
    result = multiply(time_array, freq_array)

    expected_val = jnp.array([2.0, 6.0, 12.0])
    assert jnp.allclose(result.value(), expected_val)
    assert result.unit == {}  # dimensionless


def test_multiply_overload_jax_jax():
    """Test multiplication of two jax Arrays"""
    arr1 = jnp.array([1.0, 2.0, 3.0])
    arr2 = jnp.array([4.0, 5.0, 6.0])

    result = multiply(arr1, arr2)

    expected = jnp.array([4.0, 10.0, 18.0])
    assert jnp.allclose(result, expected)
    assert isinstance(result, jax.Array)


def test_multiply_overload_numpy_numpy():
    """Test multiplication of two numpy arrays"""
    arr1 = np.array([1.0, 2.0, 3.0])
    arr2 = np.array([4.0, 5.0, 6.0])

    result = multiply(arr1, arr2)

    expected = np.array([4.0, 10.0, 18.0])
    assert np.allclose(result, expected)
    assert isinstance(result, np.ndarray)


def test_multiply_python_scalars_promotion():
    """Test multiplication with Python Scalar int, float and complex"""
    assert multiply(2, 3.5) == 7.0
    assert multiply(2, 1 + 2j) == 2 + 4j
    assert multiply(2.0, 1 - 1j) == 2 - 2j
    assert multiply(2, 3) == 6
    assert multiply(2.0, 3.0) == 6.0
    assert multiply(1 + 2j, 3 - 1j) == (5 + 5j)


# mixed type:
# unitful-unitful with val jax-numpy -> should return jax type
def test_multiply_unitful_unitful_mixed_jax_numpy_backend_priority():
    """Test multiplication of Unitful objects with value Jax Array and numpy Array"""
    a = s * jnp.array([1.0, 2.0])
    b = s * np.array([3.0, 4.0])
    res1 = multiply(a, b)
    res2 = multiply(b, a)
    assert isinstance(res1.val, jax.Array)
    assert isinstance(res2.val, jax.Array)
    assert jnp.allclose(res1.value(), jnp.array([3.0, 8.0]))
    assert jnp.allclose(res2.value(), jnp.array([3.0, 8.0]))
    assert res1.unit == {SI.s: 2}
    assert res2.unit == {SI.s: 2}


# unitful-unitful with val jax-python -> should return jax type
def test_multiply_unitful_unitful_mixed_jax_python_backend_priority():
    """Test multiplication of Unitful objects with value Jax Array and Python scalar"""
    a = s * jnp.array([2.0, 3.0])
    b = 5 * s  # Python value
    res1 = multiply(a, b)
    res2 = multiply(b, a)
    assert isinstance(res1.val, jax.Array)
    assert isinstance(res2.val, jax.Array)
    assert jnp.allclose(res1.value(), jnp.array([10.0, 15.0]))
    assert jnp.allclose(res2.value(), jnp.array([10.0, 15.0]))
    assert res1.unit == {SI.s: 2}
    assert res2.unit == {SI.s: 2}


# unitful-unitful with val numpy-python -> should return numpy type
def test_multiply_unitful_unitful_mixed_numpy_python_backend_priority():
    """Test multiplication of Unitful objects with value numpy array and Python scalar"""
    a = s * np.array([2.0, 3.0])
    b = s * 4
    res = multiply(a, b)
    assert isinstance(res.val, np.ndarray)
    assert jnp.allclose(res.value(), jnp.array([8.0, 12.0]))
    assert res.unit == {SI.s: 2}


def test_multiply_arraylike_unitful():
    """Test multiplication of ArrayLike with Unitful"""
    scalar = 5.0
    time_val = 3 * s

    result: Unitful = multiply(scalar, time_val)

    assert jnp.allclose(result.value(), 15.0)
    assert result.unit == {SI.s: 1}


# jax array - numpy array
def test_multiply_jaxarray_numpy_array_both_orders():
    """Test multiplication with regular Jax Array and Numpy Array"""
    x = jnp.array([1.0, 2.0, 3.0])
    y = np.array([4.0, 5.0, 6.0])

    r1 = multiply(x, y)
    r2 = multiply(y, x)

    assert isinstance(r1, jax.Array)
    assert isinstance(r2, jax.Array)
    expected = jnp.array([4.0, 10.0, 18.0])
    assert jnp.allclose(r1, expected)
    assert jnp.allclose(r2, expected)


# jax array - python scalar
def test_multiply_jaxarray_python_scalar_both_orders():
    """Test multiplication with regular Jax Array and Python Scalar"""
    x = jnp.array([1.0, 2.0, 3.0])
    s1 = 2.5

    r1 = multiply(x, s1)
    r2 = multiply(s1, x)

    assert isinstance(r1, jax.Array)
    assert isinstance(r2, jax.Array)
    expected = jnp.array([2.5, 5.0, 7.5])
    assert jnp.allclose(r1, expected)
    assert jnp.allclose(r2, expected)


# numpy array - python scalar
def test_multiply_numpy_array_python_scalar_both_orders():
    """Test multiplication with regular Numpy Array and Python Scalar"""
    a = np.array([2.0, 3.0])
    s1 = 4.0

    r1 = multiply(a, s1)
    r2 = multiply(s1, a)

    assert isinstance(r1, np.ndarray)
    assert isinstance(r2, np.ndarray)
    expected = np.array([8.0, 12.0])
    assert jnp.allclose(r1, expected)
    assert jnp.allclose(r2, expected)


# numpy number - python scalar
def test_multiply_numpy_scalar_python_scalar_both_orders():
    """Test multiplication with regular Numpy Number and Python Scalar"""
    ns = np.float64(2.5)
    py = 4

    r1 = multiply(ns, py)
    r2 = multiply(py, ns)

    assert isinstance(r1, np.generic)
    assert isinstance(r2, np.generic)
    assert r1 == np.float64(10.0)
    assert r2 == np.float64(10.0)


# numpy number - python scalar
def test_multiply_numpy_scalar_with_jax_both_orders():
    """Test multiplication with regular Numpy Number and Jax Array"""
    ns = np.int64(3)
    x = jnp.array([5.0, 6.0])

    r1 = multiply(ns, x)
    r2 = multiply(x, ns)

    assert isinstance(r1, jax.Array)
    assert isinstance(r2, jax.Array)
    assert jnp.allclose(r1, jnp.array([15.0, 18.0]))
    assert jnp.allclose(r2, jnp.array([15.0, 18.0]))


def test_multiply_not_implemented():
    """Test that multiply raises NotFoundLookupError for unsupported types"""
    with pytest.raises(Exception):
        multiply("string", "string")  # type: ignore


def test_multiply_jitted():
    """Test jitting of multplication"""
    arr1 = s * jnp.array([1.0, 2.0, 3.0])
    arr2 = ms * jnp.array([1.0, 2.0, 3.0])
    arr3 = ms * jnp.array([1.0, 2.0, 3.0])

    def fn(a: Unitful, b: Unitful) -> Unitful:
        return multiply(a, b)

    jitted_fn = jax.jit(fn)
    res1 = jitted_fn(arr1, arr2)
    res2 = jitted_fn(arr2, arr3)

    assert jnp.allclose(res1.value(), res2.value() * 1e3)


_v = 5
_w = Unitful(val=3.0)
_u = Unitful(val=jnp.ones(shape=(1,)) * 6)


def _fn(x: Unitful, y: Unitful, z: jax.Array):
    def _inner_fn(x2, y2):
        return x2 * y2

    def _inner_fn2(a):
        return a

    tmp = _v * z
    mid = x * y * tmp * _w * _u * 3
    y2 = jit(_inner_fn2)(y)
    out = jit(_inner_fn)(mid, y2)
    return out


def test_complicated_fn():
    unit = Unit({SI.m: 1, SI.s: -1})
    x = Unitful(
        val=jnp.array(
            [
                [3.0, 4.0, 0.0],
                [0.0, 6.0, 8.0],
                [-5.0, 0.0, 12.0],
            ]
        ),
        unit=unit,
        scale=2,
    )
    y = Unitful(val=25.0, unit=unit, scale=0)
    z = jnp.asarray([100.0])
    res = jit(_fn)(x, y, z)
    res_no_jit = _fn(x, y, z)
    scale_diff = res.scale - res_no_jit.scale
    assert jnp.allclose(res.val, res_no_jit.val * (10 ** (scale_diff)))
