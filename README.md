# Quantax: Arrays with Quantities in JAX

## Installation

TODO: You can install Quantax simply via 

```bash
pip install quantax
```
We recommend installing also the GPU-acceleration from JAX, which will massively increase speed:
```bash
pip install jax[cuda]
```

## Architecture Design

There are two modes for execution: eager and traced. 
Eager mode is used for numpy / python values or jax Array outside of jit context.
The traced mode is used only for Unitful arrays with jax Array content inside of jit.

### Eager mode
Scale of all Unitfuls is automaticallly optimized to be always optimal. 
This can easily be done since all the values are known.

### Traced mode
In the traced mode, the value of traced jax arrays is not known, so this is more complicated. 
Quantax implements its own tracing mechanism with the following characteristics:
- For Unitfuls with jax-Array as content, a tracer is used before jit to record a computational graph. The scale within the graph is optimized and afterwards the correct jax functions are replayed inside the original jax.jit to get the compiled function.
- For each jax operation, constraints on the inputs are collected (for example in addition inputs and outputs need to have same scale)
- In between operations, scale conversion can be performed to adhere to constraints, but this is optimized to minimize conversions and keeping numerical accuracy.
- During tracing, all non-jax unitful arrays are executed eagerly. When operations involve tracers and non-traced Unitfuls, non-traced values are converted as constant input nodes for the computation graph
- Unitfuls with jax Array as inputs have a fixed scale (which can be optimized because input values are known). Jax arrays with all zero-values have an unknown scale.
- Non-unitful jax arrays are converted to Unitful jax array (with unit of None) and hence traced. The unit of None is different from an empty SI-Unit, because this allows us to identify which values need to be materialised at the end.
- For small jax arrays, during compilation the traced operations are executed eagerly to better determine the optimal scale. After tracing, these operations are replayed with the jitted input.
- For complex functions, like jit/cond/while/etc. that involve function transformations, a tree of these nodes is build when they are used in a nested way. During function replay, first the innermost functions are resolved and the propagated to the outer functions.

## Development Guide
If you want to add a new function to this library, the following steps should be followed:
- 

