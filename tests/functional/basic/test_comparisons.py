import operator as op

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from quantax.functional.numpy.comparisons import array_equal, eq, ge, gt, le, lt, ne
from quantax.unitful.unitful import Unitful
from quantax.units import Hz, ms, s, us

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CMP_FNS = [
    ("eq", eq),
    ("ne", ne),
    ("lt", lt),
    ("le", le),
    ("gt", gt),
    ("ge", ge),
]

CMP_FN_IDS = [name for name, _ in CMP_FNS]


def _expected(fn_name: str, a_val: float, b_val: float) -> bool:
    """Return the expected bool result for a comparison given plain Python values."""
    return {
        "eq": op.eq,
        "ne": op.ne,
        "lt": op.lt,
        "le": op.le,
        "gt": op.gt,
        "ge": op.ge,
    }[fn_name](a_val, b_val)


# ---------------------------------------------------------------------------
# 1. Same unit, same scale – scalar values
# ---------------------------------------------------------------------------

_SCALAR_PAIRS = [
    (2.0, 3.0),  # a < b
    (3.0, 3.0),  # a == b
    (4.0, 3.0),  # a > b
]

_SCALAR_PAIR_IDS = ["a_lt_b", "a_eq_b", "a_gt_b"]


@pytest.mark.parametrize("fn_name,fn", CMP_FNS, ids=CMP_FN_IDS)
@pytest.mark.parametrize("a_val,b_val", _SCALAR_PAIRS, ids=_SCALAR_PAIR_IDS)
def test_same_unit_same_scale_scalar(fn_name, fn, a_val, b_val):
    """All six comparisons with same unit / same scale Python-float scalars."""
    a = a_val * s
    b = b_val * s
    expected = _expected(fn_name, a_val, b_val)
    assert bool(fn(a, b)) is expected


# ---------------------------------------------------------------------------
# 2. Same unit, different scales
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fn_name,fn", CMP_FNS, ids=CMP_FN_IDS)
def test_same_unit_different_scales(fn_name, fn):
    """
    1 s == 1000 ms == 1_000_000 us.
    After scale normalisation all three represent the same physical quantity.
    """
    one_s = 1 * s
    one_ks = 1000 * ms  # same as 1 s
    half_s = 500 * ms  # 0.5 s  < 1 s

    # equal pair
    expected_eq = _expected(fn_name, 1.0, 1.0)
    assert bool(fn(one_s, one_ks)) is expected_eq
    assert bool(fn(one_ks, one_s)) is expected_eq

    # unequal pair: half_s (0.5) vs one_s (1.0)
    expected_lt = _expected(fn_name, 0.5, 1.0)
    assert bool(fn(half_s, one_s)) is expected_lt


@pytest.mark.parametrize("fn_name,fn", CMP_FNS, ids=CMP_FN_IDS)
def test_same_unit_three_scales(fn_name, fn):
    """Microsecond scale included – 1 s, 1000 ms, 1_000_000 us must all be equal."""
    one_s = 1 * s
    one_ms = 1_000 * ms
    one_us = 1_000_000 * us
    expected = _expected(fn_name, 1.0, 1.0)
    assert bool(fn(one_s, one_ms)) is expected
    assert bool(fn(one_ms, one_us)) is expected
    assert bool(fn(one_s, one_us)) is expected


# ---------------------------------------------------------------------------
# 3. Different units must raise ValueError
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fn_name,fn", CMP_FNS, ids=CMP_FN_IDS)
def test_different_units_raises(fn_name, fn):
    """Comparing incompatible units must always raise ValueError."""
    with pytest.raises(Exception, match="different units"):
        fn(1 * s, 1 * Hz)


# ---------------------------------------------------------------------------
# 4. Python int / float scalar values (Unitful wrapping int & float)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fn_name,fn", CMP_FNS, ids=CMP_FN_IDS)
@pytest.mark.parametrize(
    "a_raw,b_raw",
    [(2, 3), (3, 3), (4, 3), (2.0, 3.5), (3.5, 3.5)],
    ids=["int_lt", "int_eq", "int_gt", "float_lt", "float_eq"],
)
def test_python_scalar_types(fn_name, fn, a_raw, b_raw):
    """int and float Python scalars wrapped in Unitful should compare correctly."""
    a = a_raw * s
    b = b_raw * s
    expected = _expected(fn_name, float(a_raw), float(b_raw))
    assert bool(fn(a, b)) is expected
    assert isinstance(a.val, (int, float))
    assert isinstance(b.val, (int, float))


# ---------------------------------------------------------------------------
# 5. numpy array operands
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fn_name,fn", CMP_FNS, ids=CMP_FN_IDS)
def test_numpy_arrays_elementwise(fn_name, fn):
    """Element-wise comparison of numpy-backed Unitful arrays."""
    a = s * np.array([1.0, 2.0, 3.0])
    b = s * np.array([2.0, 2.0, 2.0])
    result = fn(a, b)
    expected = np.array([_expected(fn_name, av, bv) for av, bv in zip([1, 2, 3], [2, 2, 2])])
    assert array_equal(result, expected)
    # result should remain in numpy territory (not promoted to jax)
    assert isinstance(result, Unitful)


@pytest.mark.parametrize("fn_name,fn", CMP_FNS, ids=CMP_FN_IDS)
def test_numpy_arrays_different_scales_elementwise(fn_name, fn):
    """Element-wise comparison of numpy-backed Unitful arrays with mixed scales."""
    # a in seconds, b in milliseconds – scale normalisation must happen
    a = s * np.array([1.0, 0.5, 2.0])  # 1 s, 0.5 s, 2 s
    b = ms * np.array([1000.0, 1000.0, 500.0])  # 1 s, 1 s,   0.5 s
    result = fn(a, b)
    expected = np.array(
        [
            _expected(fn_name, 1.0, 1.0),
            _expected(fn_name, 0.5, 1.0),
            _expected(fn_name, 2.0, 0.5),
        ]
    )
    assert array_equal(result, expected)


# ---------------------------------------------------------------------------
# 6. JAX array operands
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fn_name,fn", CMP_FNS, ids=CMP_FN_IDS)
def test_jax_arrays_elementwise(fn_name, fn):
    """Element-wise comparison of JAX-backed Unitful arrays."""
    a = s * jnp.array([1.0, 2.0, 3.0])
    b = s * jnp.array([2.0, 2.0, 2.0])
    result = fn(a, b)
    expected = jnp.array([_expected(fn_name, av, bv) for av, bv in zip([1, 2, 3], [2, 2, 2])])
    assert array_equal(result, expected)
    assert isinstance(result, Unitful)


@pytest.mark.parametrize("fn_name,fn", CMP_FNS, ids=CMP_FN_IDS)
def test_jax_arrays_different_scales_elementwise(fn_name, fn):
    """Element-wise comparison of JAX-backed Unitful arrays with mixed scales."""
    a = s * jnp.array([1.0, 0.5, 2.0])
    b = ms * jnp.array([1000.0, 1000.0, 500.0])
    result = fn(a, b)
    expected = jnp.array(
        [
            _expected(fn_name, 1.0, 1.0),
            _expected(fn_name, 0.5, 1.0),
            _expected(fn_name, 2.0, 0.5),
        ]
    )
    assert array_equal(result, expected)


# ---------------------------------------------------------------------------
# 7. Mixed backend: JAX array + numpy array (both argument orders)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fn_name,fn", CMP_FNS, ids=CMP_FN_IDS)
def test_mixed_jax_numpy_backend_both_orders(fn_name, fn):
    """Unitful JAX array vs Unitful numpy array – result should be a JAX array."""
    a = s * jnp.array([1.0, 2.0, 3.0])  # jax-backed
    b = s * np.array([2.0, 2.0, 2.0])  # numpy-backed
    expected_r1 = jnp.array([_expected(fn_name, av, bv) for av, bv in zip([1, 2, 3], [2, 2, 2])])
    expected_r2 = jnp.array([_expected(fn_name, bv, av) for av, bv in zip([1, 2, 3], [2, 2, 2])])

    r1 = fn(a, b)
    r2 = fn(b, a)

    assert isinstance(r1, Unitful)
    assert isinstance(r2, Unitful)
    assert array_equal(r1, expected_r1)
    assert array_equal(r2, expected_r2)


# ---------------------------------------------------------------------------
# 8. Mixed backend: JAX array + Python scalar (both argument orders)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fn_name,fn", CMP_FNS, ids=CMP_FN_IDS)
def test_mixed_jax_python_backend_both_orders(fn_name, fn):
    """Unitful JAX array vs Unitful Python scalar – result should be a JAX array."""
    a = s * jnp.array([1.0, 2.0, 3.0])
    b = 2 * s  # Python int scalar

    r1 = fn(a, b)
    r2 = fn(b, a)

    assert isinstance(r1, Unitful)
    assert isinstance(r2, Unitful)

    expected_r1 = jnp.array([_expected(fn_name, av, 2.0) for av in [1, 2, 3]])
    expected_r2 = jnp.array([_expected(fn_name, 2.0, av) for av in [1, 2, 3]])
    assert array_equal(r1, expected_r1)
    assert array_equal(r2, expected_r2)


# ---------------------------------------------------------------------------
# 9. Mixed backend: numpy array + Python scalar (both argument orders)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fn_name,fn", CMP_FNS, ids=CMP_FN_IDS)
def test_mixed_numpy_python_backend_both_orders(fn_name, fn):
    """Unitful numpy array vs Unitful Python scalar – result should stay numpy."""
    a = s * np.array([1.0, 2.0, 3.0])
    b = 2 * s

    r1 = fn(a, b)
    r2 = fn(b, a)

    assert isinstance(r1, Unitful)
    assert isinstance(r2, Unitful)

    expected_r1 = np.array([_expected(fn_name, av, 2.0) for av in [1, 2, 3]])
    expected_r2 = np.array([_expected(fn_name, 2.0, av) for av in [1, 2, 3]])
    assert array_equal(r1, expected_r1)
    assert array_equal(r2, expected_r2)


# ---------------------------------------------------------------------------
# 10. numpy scalar (np.generic) operands
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fn_name,fn", CMP_FNS, ids=CMP_FN_IDS)
def test_numpy_scalar_unitful(fn_name, fn):
    """numpy scalar-valued Unitful objects should compare correctly."""
    a = np.float64(2.0) * s
    b = np.float64(3.0) * s
    assert bool(fn(a, b)) is _expected(fn_name, 2.0, 3.0)


@pytest.mark.parametrize("fn_name,fn", CMP_FNS, ids=CMP_FN_IDS)
def test_numpy_scalar_unitful_vs_python_scalar_unitful(fn_name, fn):
    """numpy scalar Unitful vs Python scalar Unitful – both orders."""
    ns = np.float64(2.5) * s
    py = 3 * s
    r1 = fn(ns, py)
    r2 = fn(py, ns)
    assert bool(r1) is _expected(fn_name, 2.5, 3.0)
    assert bool(r2) is _expected(fn_name, 3.0, 2.5)


@pytest.mark.parametrize("fn_name,fn", CMP_FNS, ids=CMP_FN_IDS)
def test_numpy_scalar_unitful_vs_jax_array_unitful(fn_name, fn):
    """numpy scalar Unitful vs JAX array Unitful – result should be JAX array."""
    ns = np.int64(2) * s
    jx = s * jnp.array([1.0, 2.0, 3.0])
    r1 = fn(ns, jx)
    r2 = fn(jx, ns)
    assert isinstance(r1, Unitful)
    assert isinstance(r2, Unitful)
    exp_r1 = jnp.array([_expected(fn_name, 2.0, av) for av in [1, 2, 3]])
    exp_r2 = jnp.array([_expected(fn_name, av, 2.0) for av in [1, 2, 3]])
    assert array_equal(r1, exp_r1)
    assert array_equal(r2, exp_r2)


# ---------------------------------------------------------------------------
# 11. Plain (non-Unitful) fallback path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fn_name,fn", CMP_FNS, ids=CMP_FN_IDS)
@pytest.mark.parametrize(
    "a_val,b_val",
    [(2, 3), (3, 3), (4, 3)],
    ids=["lt", "eq", "gt"],
)
def test_plain_python_scalar_fallback(fn_name, fn, a_val, b_val):
    """Plain Python int scalars go through the non-Unitful fallback."""
    assert bool(fn(a_val, b_val)) is _expected(fn_name, float(a_val), float(b_val))


@pytest.mark.parametrize("fn_name,fn", CMP_FNS, ids=CMP_FN_IDS)
def test_plain_jax_array_fallback(fn_name, fn):
    """Plain JAX arrays (no unit) go through the non-Unitful fallback."""
    a = jnp.array([1.0, 2.0, 3.0])
    b = jnp.array([2.0, 2.0, 2.0])
    result = fn(a, b)
    expected = jnp.array([_expected(fn_name, av, bv) for av, bv in zip([1, 2, 3], [2, 2, 2])])
    assert jnp.array_equal(result, expected)
    assert isinstance(result, jax.Array)


@pytest.mark.parametrize("fn_name,fn", CMP_FNS, ids=CMP_FN_IDS)
def test_plain_numpy_array_fallback(fn_name, fn):
    """Plain numpy arrays (no unit) go through the non-Unitful fallback."""
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([2.0, 2.0, 2.0])
    result = fn(a, b)
    expected = np.array([_expected(fn_name, av, bv) for av, bv in zip([1, 2, 3], [2, 2, 2])])
    assert np.array_equal(result, expected)
    assert isinstance(result, (np.ndarray, np.bool_))


@pytest.mark.parametrize("fn_name,fn", CMP_FNS, ids=CMP_FN_IDS)
def test_plain_jax_numpy_mixed_fallback_both_orders(fn_name, fn):
    """Plain JAX + numpy arrays (no unit) – JAX should win."""
    a = jnp.array([1.0, 2.0, 3.0])
    b = np.array([2.0, 2.0, 2.0])
    expected_r1 = jnp.array([_expected(fn_name, av, bv) for av, bv in zip([1, 2, 3], [2, 2, 2])])
    expected_r2 = jnp.array([_expected(fn_name, bv, av) for av, bv in zip([1, 2, 3], [2, 2, 2])])
    r1 = fn(a, b)
    r2 = fn(b, a)
    assert isinstance(r1, jax.Array)
    assert isinstance(r2, jax.Array)
    assert array_equal(r1, expected_r1)
    assert array_equal(r2, expected_r2)


# ---------------------------------------------------------------------------
# 12. Unitful vs plain (dimensionless) interop
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fn_name,fn", CMP_FNS, ids=CMP_FN_IDS)
def test_unitful_dimensionless_vs_plain_scalar(fn_name, fn):
    """Dimensionless Unitful vs bare Python scalar should work via fallback."""
    a = Unitful(val=5.0)
    result = fn(a, 5.0)
    assert bool(result) is _expected(fn_name, 5.0, 5.0)


@pytest.mark.parametrize("fn_name,fn", CMP_FNS, ids=CMP_FN_IDS)
def test_unitful_dimensionless_vs_plain_jax_array(fn_name, fn):
    """Dimensionless Unitful array vs bare JAX array."""
    a = Unitful(val=jnp.array([1.0, 2.0, 3.0]))
    b = jnp.array([2.0, 2.0, 2.0])
    result = fn(a, b)
    expected = jnp.array([_expected(fn_name, av, bv) for av, bv in zip([1, 2, 3], [2, 2, 2])])
    assert array_equal(result, expected)


# ---------------------------------------------------------------------------
# 13. Magic-method dunder operators (__eq__, __ne__, __lt__, __le__, __gt__, __ge__)
# ---------------------------------------------------------------------------

_MAGIC_PARAMS = [
    ("eq", "__eq__"),
    ("ne", "__ne__"),
    ("lt", "__lt__"),
    ("le", "__le__"),
    ("gt", "__gt__"),
    ("ge", "__ge__"),
]


@pytest.mark.parametrize("fn_name,magic", _MAGIC_PARAMS, ids=CMP_FN_IDS)
@pytest.mark.parametrize("a_val,b_val", _SCALAR_PAIRS, ids=_SCALAR_PAIR_IDS)
def test_magic_method_scalar(fn_name, magic, a_val, b_val):
    """Dunder methods on Unitful must agree with the functional API."""
    a = a_val * s
    b = b_val * s
    expected = _expected(fn_name, a_val, b_val)
    assert bool(getattr(a, magic)(b)) is expected


@pytest.mark.parametrize("fn_name,magic", _MAGIC_PARAMS, ids=CMP_FN_IDS)
def test_magic_method_arrays(fn_name, magic):
    """Dunder methods should work element-wise on JAX-array-backed Unitful."""
    a = s * jnp.array([1.0, 2.0, 3.0])
    b = s * jnp.array([2.0, 2.0, 2.0])
    result = getattr(a, magic)(b)
    expected = jnp.array([_expected(fn_name, av, bv) for av, bv in zip([1, 2, 3], [2, 2, 2])])
    assert array_equal(result, expected)


# ---------------------------------------------------------------------------
# 14. JIT compilation – all six operators
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fn_name,fn", CMP_FNS, ids=CMP_FN_IDS)
def test_jit_scalar(fn_name, fn):
    """All comparisons survive jax.jit with scalar Unitful operands."""
    a = 2.0 * s
    b = 3.0 * s
    jitted = jax.jit(fn)
    assert bool(jitted(a, b)) is _expected(fn_name, 2.0, 3.0)


@pytest.mark.parametrize("fn_name,fn", CMP_FNS, ids=CMP_FN_IDS)
def test_jit_array(fn_name, fn):
    """All comparisons survive jax.jit with JAX-array Unitful operands."""
    a = s * jnp.array([1.0, 2.0, 3.0])
    b = s * jnp.array([2.0, 2.0, 2.0])
    jitted = jax.jit(fn)
    result = jitted(a, b)
    expected = jnp.array([_expected(fn_name, av, bv) for av, bv in zip([1, 2, 3], [2, 2, 2])])
    assert array_equal(result, expected)


@pytest.mark.parametrize("fn_name,fn", CMP_FNS, ids=CMP_FN_IDS)
def test_jit_different_scales(fn_name, fn):
    """JIT must still normalise scales before comparing."""
    a = s * jnp.array([1.0, 0.5])
    b = ms * jnp.array([1000.0, 1000.0])  # 1 s, 1 s
    jitted = jax.jit(fn)
    result = jitted(a, b)
    expected = jnp.array(
        [
            _expected(fn_name, 1.0, 1.0),
            _expected(fn_name, 0.5, 1.0),
        ]
    )
    assert array_equal(result, expected)


@pytest.mark.parametrize("fn_name,fn", CMP_FNS, ids=CMP_FN_IDS)
def test_jit_reuse(fn_name, fn):
    """Jitted function should be reusable across multiple calls without retracing issues."""
    jitted = jax.jit(fn)
    for a_val, b_val in [(1.0, 2.0), (3.0, 3.0), (5.0, 4.0)]:
        a = a_val * s
        b = b_val * s
        assert bool(jitted(a, b)) is _expected(fn_name, a_val, b_val)


# ---------------------------------------------------------------------------
# 15. Edge cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fn_name,fn", CMP_FNS, ids=CMP_FN_IDS)
def test_zero_values(fn_name, fn):
    """Zero-valued Unitful scalars should compare correctly."""
    a = 0.0 * s
    b = 0.0 * s
    assert bool(fn(a, b)) is _expected(fn_name, 0.0, 0.0)


@pytest.mark.parametrize("fn_name,fn", CMP_FNS, ids=CMP_FN_IDS)
def test_negative_values(fn_name, fn):
    """Negative-valued Unitful scalars should compare correctly."""
    a = -3.0 * s
    b = -1.0 * s
    assert bool(fn(a, b)) is _expected(fn_name, -3.0, -1.0)


@pytest.mark.parametrize("fn_name,fn", CMP_FNS, ids=CMP_FN_IDS)
def test_large_array(fn_name, fn):
    """Comparison over a larger JAX array should be element-wise and correct."""
    rng = np.random.default_rng(42)
    a_vals = rng.uniform(-10, 10, size=(100,))
    b_vals = rng.uniform(-10, 10, size=(100,))
    a = s * jnp.array(a_vals)
    b = s * jnp.array(b_vals)
    result = fn(a, b)
    expected = jnp.array([_expected(fn_name, float(av), float(bv)) for av, bv in zip(a_vals, b_vals)])
    assert jnp.array_equal(result, expected)


@pytest.mark.parametrize("fn_name,fn", CMP_FNS, ids=CMP_FN_IDS)
def test_2d_array(fn_name, fn):
    """Comparison works on 2-D JAX arrays."""
    a = s * jnp.array([[1.0, 2.0], [3.0, 4.0]])
    b = s * jnp.array([[2.0, 2.0], [2.0, 5.0]])
    result = fn(a, b)
    expected = jnp.array(
        [
            [_expected(fn_name, 1.0, 2.0), _expected(fn_name, 2.0, 2.0)],
            [_expected(fn_name, 3.0, 2.0), _expected(fn_name, 4.0, 5.0)],
        ]
    )
    assert jnp.array_equal(result, expected)
