import jax.numpy as jnp
from jax import core


def is_currently_compiling() -> bool:
    return isinstance(
        jnp.array(1) + 1,
        core.Tracer,
    )


def is_traced(x) -> bool:
    return isinstance(x, core.Tracer)
