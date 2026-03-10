import jax
import jax.numpy as jnp
import numpy as np
import pytest

from quantax.core.typing import SI
from quantax.functional.numpy.comparisons import eq, ge, gt, le, lt, ne
from quantax.unitful.unitful import Unitful
from quantax.units import Hz, ms, s


## eq ###########################


def test_eq_same_unit_same_scale_true():
    a = 3 * s
    b = 3 * s
    assert bool(eq(a, b)) is True


def test_eq_same_unit_same_scale_false():
    a = 3 * s
    b = 4 * s
    assert bool(eq(a, b)) is False


def test_eq_same_unit_different_scales():
    a = 1 * s  # 1e0 s
    b = 1000 * ms  # 1e-3 * 1000 = 1 s
    assert bool(eq(a, b)) is True


def test_eq_different_units_raises():
    with pytest.raises(ValueError):
        eq(1 * s, 1 * Hz)


def test_eq_arrays():
    a = s * jnp.array([1.0, 2.0, 3.0])
    b = s * jnp.array([1.0, 0.0, 3.0])
    result = eq(a, b)
    assert jnp.array_equal(result, jnp.array([True, False, True]))


def test_eq_unitful_vs_plain():
    a = Unitful(val=5.0)
    result = eq(a, 5.0)
    assert bool(result) is True


def test_eq_plain_fallback():
    result = eq(jnp.array([1.0, 2.0]), jnp.array([1.0, 3.0]))
    assert jnp.array_equal(result, jnp.array([True, False]))


def test_eq_magic_method():
    a = 2 * s
    b = 2 * s
    assert bool(a == b) is True


## ne ###########################


def test_ne_basic():
    a = 3 * s
    b = 4 * s
    assert bool(ne(a, b)) is True


def test_ne_equal_values():
    a = 3 * s
    b = 3 * s
    assert bool(ne(a, b)) is False


def test_ne_different_units_raises():
    with pytest.raises(ValueError):
        ne(1 * s, 1 * Hz)


def test_ne_magic_method():
    a = 2 * s
    b = 3 * s
    assert bool(a != b) is True


## lt ###########################


def test_lt_true():
    a = 2 * s
    b = 3 * s
    assert bool(lt(a, b)) is True


def test_lt_false():
    a = 3 * s
    b = 2 * s
    assert bool(lt(a, b)) is False


def test_lt_same_unit_different_scales():
    a = 1 * ms  # 0.001 s
    b = 1 * s   # 1 s
    assert bool(lt(a, b)) is True


def test_lt_different_units_raises():
    with pytest.raises(ValueError):
        lt(1 * s, 1 * Hz)


def test_lt_magic_method():
    a = 1 * s
    b = 2 * s
    assert bool(a < b) is True


## le ###########################


def test_le_less():
    assert bool(le(2 * s, 3 * s)) is True


def test_le_equal():
    assert bool(le(3 * s, 3 * s)) is True


def test_le_greater():
    assert bool(le(4 * s, 3 * s)) is False


def test_le_different_units_raises():
    with pytest.raises(ValueError):
        le(1 * s, 1 * Hz)


def test_le_magic_method():
    a = 2 * s
    b = 2 * s
    assert bool(a <= b) is True


## gt ###########################


def test_gt_true():
    assert bool(gt(3 * s, 2 * s)) is True


def test_gt_false():
    assert bool(gt(2 * s, 3 * s)) is False


def test_gt_different_units_raises():
    with pytest.raises(ValueError):
        gt(1 * s, 1 * Hz)


def test_gt_magic_method():
    a = 3 * s
    b = 2 * s
    assert bool(a > b) is True


## ge ###########################


def test_ge_greater():
    assert bool(ge(3 * s, 2 * s)) is True


def test_ge_equal():
    assert bool(ge(3 * s, 3 * s)) is True


def test_ge_less():
    assert bool(ge(2 * s, 3 * s)) is False


def test_ge_different_units_raises():
    with pytest.raises(ValueError):
        ge(1 * s, 1 * Hz)


def test_ge_magic_method():
    a = 3 * s
    b = 2 * s
    assert bool(a >= b) is True


## Numpy arrays ###########################


def test_comparisons_numpy_arrays():
    a = s * np.array([1.0, 2.0, 3.0])
    b = s * np.array([2.0, 2.0, 2.0])
    assert np.array_equal(lt(a, b), np.array([True, False, False]))
    assert np.array_equal(le(a, b), np.array([True, True, False]))
    assert np.array_equal(gt(a, b), np.array([False, False, True]))
    assert np.array_equal(ge(a, b), np.array([False, True, True]))
    assert np.array_equal(eq(a, b), np.array([False, True, False]))
    assert np.array_equal(ne(a, b), np.array([True, False, True]))


## Plain scalar fallback ###########################


def test_plain_scalar_fallback():
    assert bool(eq(3, 3)) is True
    assert bool(ne(3, 4)) is True
    assert bool(lt(2, 3)) is True
    assert bool(le(3, 3)) is True
    assert bool(gt(4, 3)) is True
    assert bool(ge(3, 3)) is True


## Jitted execution ###########################


def test_eq_jitted():
    a = s * jnp.array([1.0, 2.0, 3.0])
    b = s * jnp.array([1.0, 0.0, 3.0])

    def fn(x, y):
        return eq(x, y)

    result = jax.jit(fn)(a, b)
    assert jnp.array_equal(result, jnp.array([True, False, True]))


def test_lt_jitted():
    a = s * jnp.array([1.0, 2.0, 3.0])
    b = s * jnp.array([2.0, 2.0, 2.0])

    def fn(x, y):
        return lt(x, y)

    result = jax.jit(fn)(a, b)
    assert jnp.array_equal(result, jnp.array([True, False, False]))


def test_comparisons_scale_normalization():
    """Test that scale differences are properly normalized before comparison."""
    # 1000 ms == 1 s
    a = 1000 * ms
    b = 1 * s
    assert bool(eq(a, b)) is True
    assert bool(ne(a, b)) is False
    assert bool(le(a, b)) is True
    assert bool(ge(a, b)) is True

    # 500 ms < 1 s
    c = 500 * ms
    assert bool(lt(c, b)) is True
    assert bool(gt(c, b)) is False
