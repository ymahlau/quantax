# Contributing to quantax

## Setup

1. Fork the repository on GitHub and clone it locally.
2. Install [uv](https://docs.astral.sh/uv/getting-started/installation/).
3. Install the dev environment:
   ```bash
   uv sync --dev
   ```
4. Verify everything works:
   ```bash
   uv run pytest
   ```

---

## Adding a basic operation

Basic operations (like `multiply`, `add`, `subtract`, `divide`) wrap individual JAX functions and use `OperatorNode` to record computation steps during tracing.

### Step 1 – Implement in `src/quantax/functional/numpy/basic.py`

Each operation requires three pieces:

**1a. Scale constraint function**

Returns a list of OR-Tools `BoundedLinearExpression` constraints that relate the input and output scale variables. These are used by the MILP optimizer.

```python
def constraints_<name>(
    x: mathopt.Variable | None,
    y: mathopt.Variable | None,
    out: Any | None,
) -> list[mathopt.BoundedLinearExpression]:
    if out is None:
        assert x is None and y is None
        return []
    assert isinstance(out, mathopt.LinearSum)
    if x is None:
        return [y == out]   # adjust for your operation
    if y is None:
        return [x == out]
    return [x + y == out]   # adjust for your operation
```

Scale constraint reference by operation:
- `multiply`: `x + y == out`
- `add` / `subtract`: `x == out` and `y == out`
- `divide`: `x - y == out`

**1b. Original function accessor**

Retrieves the unpatched JAX function (handles the case where it has already been monkey-patched):

```python
def get_<name>_original():
    if hasattr(jax.numpy, "_orig_<name>"):
        return jax.numpy._orig_<name>
    return jax.numpy.<name>
```

**1c. The function itself**

Dispatch on input types using `isinstance`:

```python
def <name>(x: AnyUnitType, y: AnyUnitType) -> AnyUnitType:
    # --- Unitful eager case ---
    def _<name>_unitful(x: Unitful, y: Unitful):
        # compute new_unit, new_val, new_scale
        return Unitful(val=new_val, unit=new_unit, scale=new_scale)

    if isinstance(x, Unitful) and isinstance(y, Unitful):
        return _<name>_unitful(x, y)
    elif isinstance(x, Unitful) and not isinstance(y, (Unitful | UnitfulTracer)):
        return _<name>_unitful(x, Unitful(val=y))
    elif not isinstance(x, (Unitful | UnitfulTracer)) and isinstance(y, Unitful):
        return _<name>_unitful(Unitful(val=x), y)

    # --- UnitfulTracer case (inside jit) ---
    def _<name>_tracer(x: UnitfulTracer, y: UnitfulTracer):
        # compute new_unit for the result tracer
        result = UnitfulTracer(unit=new_unit, static_unitful=...)
        node = OperatorNode(
            op_name="<name>",
            op_kwargs={"x": x, "y": y},
            output=result,
        )
        register_node_full(node)
        return result

    if isinstance(x, UnitfulTracer) and isinstance(y, UnitfulTracer):
        return _<name>_tracer(x, y)
    if isinstance(x, UnitfulTracer):
        y_unitful = Unitful(val=y) if not isinstance(y, Unitful) else y
        tracer_unit = None if not isinstance(y, Unitful) else y.unit
        return _<name>_tracer(x, UnitfulTracer(unit=tracer_unit, static_unitful=y_unitful, value=y))
    if isinstance(y, UnitfulTracer):
        x_unitful = Unitful(val=x) if not isinstance(x, Unitful) else x
        tracer_unit = None if not isinstance(x, Unitful) else x.unit
        return _<name>_tracer(UnitfulTracer(unit=tracer_unit, static_unitful=x_unitful, value=x), y)

    # --- Plain array fallback ---
    assert isinstance(x, get_args(AnyArrayLike))
    assert isinstance(y, get_args(AnyArrayLike))
    return x <op> y
```

See `basic.py:21–167` for `multiply` as a complete reference.

### Step 2 – Add `@overload` signatures

Add typed overloads above the implementation for all meaningful input type combinations:

```python
@overload
def <name>(x: Unitful, y: Unitful) -> Unitful: ...

@overload
def <name>(x: AnyArrayLike, y: Unitful) -> Unitful: ...

@overload
def <name>(x: Unitful, y: AnyArrayLike) -> Unitful: ...

@overload
def <name>(x: int, y: int) -> int: ...

# ... scalars, numpy arrays, jax arrays, cross combinations
```

See `basic.py:46–112` for the full set used by `multiply`.

### Step 3 – Export from `src/quantax/functional/__init__.py`

```python
from quantax.functional.numpy.basic import <name>

__all__ = [
    ...,
    "<name>",
]
```

### Step 4a – Register in `src/quantax/functional/collection.py`

```python
from quantax.functional.numpy.basic import (
    ...,
    constraints_<name>, get_<name>_original, <name>,
)

CONSTRAINTS_DICT: dict[str, Callable] = {
    ...,
    "<name>": constraints_<name>,
}

FUNCTION_DICT: dict[str, Callable] = {
    ...,
    "<name>": <name>,
}

ORIG_FUNCTION_DICT: dict[str, Callable] = {
    ...,
    "<name>": get_<name>_original(),
}
```

### Step 4b – Register in `src/quantax/functional/patching.py`

Import your function and add it to the appropriate patch list. Pass `None` as the second element to use the function's own name, or a string to register it under a different name (e.g. `"true_divide"`):

```python
from quantax.functional.numpy.basic import <name>

_full_patch_list_numpy = [
    ...,
    (<name>, None),          # patches jax.numpy.<name>
    # (<name>, "alias"),     # patches jax.numpy.alias instead
]
```

Use `_full_patch_list_lax` for `jax.lax` functions and `_full_patch_list_linalg` for `jax.numpy.linalg`.

### Step 5 – Write tests

Create `tests/functional/basic/test_<name>.py`. Cover:

- Eager `Unitful` inputs (same and different units)
- Scale handling (inputs with different scales)
- Type preservation (Python scalar, numpy array, JAX array)
- Mixed-type inputs (JAX × numpy, Unitful × plain array)
- Fallback for plain arrays without units
- Jitted execution via `jax.jit`

If the operation is binary and should be tested against the original JAX output, also add its name to `BINARY_FNS` in `tests/functional/test_binary_functions.py`:

```python
BINARY_FNS = [
    "multiply",
    "<name>",   # add here
]
```

---

## Adding a function transform operation

Function transforms (like `jit`, `cond`, `grad`) wrap entire sub-computations. They require a `FunctionTransformNode` subclass — not a plain `OperatorNode` — and are **not** registered in `collection.py`.

The key difference from basic operations: a transform manages a local `TraceData` for its function body (`fn_tracers`) and communicates with the global computation graph via equality no-op constraints on its inputs and outputs. This allows nested transforms and handles cycles (e.g., loops).

### Structure

```
src/quantax/functional/<name>.py
```

**1. Subclass `FunctionTransformNode`**

```python
from quantax.core.glob import FunctionTransformNode

@dataclass(kw_only=True)
class <Name>TransformNode(FunctionTransformNode):
    # any extra fields your transform needs
    <name>_kwargs: dict[str, Any]

    def replay_node(self, *args, **kwargs):
        replay_data = get_global_replay_data()
        cur_graph_data = replay_data.graph_data_dict[self.id][0]
        replay_fn = get_replay_function(
            graph_data=cur_graph_data,
            trace_args=self.trace_args,
            trace_kwargs=self.trace_kwargs,
            trace_result=self.output,
        )
        return get_<name>_original()(replay_fn, **self.<name>_kwargs)(*args, **kwargs)
```

**2. Write the wrapper callable**

The callable intercepts the JAX transform. When called without an enclosing traced context (`not has_tracer`), just call the original function. When called inside an already-traced context, register the node and return tracers.

**3. Register in `patching.py` only**

Add your function to the appropriate list in `patch_all_functions_jax()`. For top-level JAX transforms, use `_full_patch_list_jax`:

```python
from quantax.functional.<name> import <name>

_full_patch_list_jax = [
    (jit, None),
    (<name>, None),
]
```

**Do not** add to `CONSTRAINTS_DICT`, `FUNCTION_DICT`, or `ORIG_FUNCTION_DICT` in `collection.py`.

### Reference implementations

- `src/quantax/functional/jit.py` — complete, working implementation. Note that jit is different because it starts tracing.
- `src/quantax/core/glob.py` — `FunctionTransformNode` definition

---

## Verification

```bash
uv run pytest tests/
uv run pre-commit run --all
```
