"""Global optimization stop flag. If True, then no arrays are statically saved. Additionally, no jax functions
like jnp.ones will return a Unitful instead of Array for scale optimization.
"""

STATIC_OPTIM_STOP_FLAG: bool = False
