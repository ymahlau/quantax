import jax.numpy as jnp
import pytest

from quantax.functional.collection import FUNCTION_DICT, ORIG_FUNCTION_DICT
from quantax.unitful.unitful import Unitful

BINARY_FNS = ["multiply"]


def get_binary_function_list_from_op(op):
    # test functions
    def _fn0(x, y):
        return op(x, y)

    def _fn1(x, y):
        return op(op(op(op(x, y), y), x), y)

    def _fn2(x, y):
        return op(op(x, y), op(x, y))

    def _fn3(x, y):
        return op(op(x, y), 1)

    def _fn4(x, y):
        return op(x, 1)

    def _fn5(x, y):
        return op(y, 1)

    def _fn6(x, y):
        return op(op(x, y), jnp.asarray(1.0))

    one = jnp.asarray(1.0)

    def _fn7(x, y):
        return op(one, op(x, y))

    return [
        _fn0,
        _fn1,
        _fn2,
        _fn3,
        _fn4,
        _fn5,
    ]


SAMPLE_INPUTS = [
    1.0,
    jnp.asarray(1.0),
    jnp.asarray([1.0, 2.0, -3.0]),
]

UNITFUL_SAMPLE_INPUTS = [
    1.0,
    jnp.asarray(1.0),
    jnp.asarray([1.0, 2.0, -3.0]),
    Unitful(val=1.0),
    Unitful(val=jnp.asarray(1.0)),
    Unitful(val=jnp.asarray([1.0, 2.0, -3.0])),
]


@pytest.mark.parametrize("op_name", BINARY_FNS)
@pytest.mark.parametrize("x", SAMPLE_INPUTS)
@pytest.mark.parametrize("y", SAMPLE_INPUTS)
def test_same_result(op_name, x, y):
    op = FUNCTION_DICT[op_name]
    orig_op = ORIG_FUNCTION_DICT[op_name]
    fn_list = get_binary_function_list_from_op(op)
    orig_fn_list = get_binary_function_list_from_op(orig_op)
    for fn, orig_fn in zip(fn_list, orig_fn_list):
        res = fn(x, y)
        orig_res = orig_fn(x, y)
        assert jnp.allclose(res, orig_res)
