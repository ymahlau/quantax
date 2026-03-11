# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Quantax is a Python library for JAX arrays with physical units and automatic scale optimization. It wraps JAX arrays in a `Unitful` type that tracks SI units and automatically chooses optimal numerical scales (powers of 10) to maintain numerical accuracy.

## Commands

```bash
# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/functional/basic/test_multiply.py

# Run a single test by name
uv run pytest tests/functional/basic/test_multiply.py::test_multiply_jitted

# Lint and auto-fix
uv run ruff check --fix src/

# Format
uv run ruff format src/

# Pre-commit checks
uv run pre-commit run --all
```

## Architecture

### Core concepts

- **`Unitful`** (`src/quantax/unitful/unitful.py`): The primary type. Holds `val` (the numeric content), `unit` (a `Unit` mapping SI enum → exponent), and `scale` (integer power of 10 offset, so actual value = `val * 10^scale`). Scale is automatically optimized in eager mode.
- **`Unit`** (`src/quantax/core/unit.py`): Immutable `frozendict[SI, int|IntFraction]` representing a physical dimension.
- **`SI`** (`src/quantax/core/typing.py`): Enum of base SI dimensions: s, m, kg, A, K, cd.
- **`units.py`** (`src/quantax/units.py`): Pre-defined unit instances (e.g. `s`, `ms`, `Hz`, `kHz`, `m`, `W`, etc.) used like `5 * ms` to create a `Unitful`.

### Execution modes

**Eager mode** (numpy/python values, or JAX arrays outside `jit`): Scale is optimized automatically on construction since all values are known.

**Traced mode** (JAX arrays inside `jit`): A two-phase system:
1. **Tracing phase**: Before the real `jit`, a pre-trace runs with `UnitfulTracer` objects instead of real arrays. This builds a global computation graph (`GlobalTraceData` in `src/quantax/core/glob.py`) of operator nodes and tracer nodes.
2. **MILP optimization**: `solve_scale_assignment()` in `src/quantax/tracing/optimization.py` formulates scale assignment as a Mixed-Integer Linear Program (using Google OR-Tools) and finds optimal integer scales for all intermediate values.
3. **Replay phase**: The optimized scales are used to replay the computation inside the real JAX `jit`, producing a compiled function with correct scales.

### Key modules

- **`src/quantax/core/glob.py`**: Global mutable state for tracing (`CURRENT_NODE`, `CURRENT_TRACE_DATA`, `GLOBAL_TRACE_DATA`, `GLOBAL_REPLAY_DATA`) and dataclasses `GlobalTraceData`, `TraceData`, `FunctionTransformNode`.
- **`src/quantax/functional/jit.py`**: Custom `jit` wrapper (`UnitfulJitWrapped`) that intercepts JAX's `jit` to perform tracing + MILP + replay when `Unitful` arrays are present.
- **`src/quantax/tracing/graph.py`**: Builds a `rustworkx.PyDiGraph` from trace data for topological ordering.
- **`src/quantax/tracing/optimization.py`**: MILP formulation via OR-Tools `mathopt` to solve optimal scale assignment.
- **`src/quantax/tracing/replay.py`**: Replays recorded operations in topological order using solved scales.
- **`src/quantax/functional/collection.py`**: Registry mapping operation names to their implementation functions (`FUNCTION_DICT`) and constraint generators (`CONSTRAINTS_DICT`).
- **`src/quantax/functional/patching.py`**: `patch_all_functions_jax()` monkey-patches `jax.numpy` and `jax.lax` with Unitful-aware versions (currently only `multiply` is active; others are commented out).

### Adding a new basic operation

Basic operations (e.g. `multiply`, `add`) live in `src/quantax/functional/numpy/basic.py`. Use `multiply` (`basic.py:21–167`) as reference.

Each operation needs:
1. `constraints_<name>()` — OR-Tools constraints relating input/output scale variables. Scale constraint reference:
   - `multiply`: `x + y == out`
   - `add` / `subtract`: `x == out` and `y == out`
   - `divide`: `x - y == out`
2. `get_<name>_original()` — returns the unpatched JAX function (checks `jax.numpy._orig_<name>` first).
3. `<name>()` — implementation dispatching on `Unitful`, `UnitfulTracer`, and plain array types.
4. `@overload` signatures for all meaningful type combinations.
5. Export from `src/quantax/functional/__init__.py`.
6. Register in `src/quantax/functional/collection.py` in `CONSTRAINTS_DICT`, `FUNCTION_DICT`, and `ORIG_FUNCTION_DICT`.
7. Register in `src/quantax/functional/patching.py` under `_full_patch_list_numpy` (or `_full_patch_list_lax` / `_full_patch_list_linalg`).
8. Write tests in `tests/functional/basic/test_<name>.py` covering: eager, scale handling, type preservation, mixed-type, plain array fallback, jitted. Add to `BINARY_FNS` in `tests/functional/test_binary_functions.py` if applicable.

### Adding a function transform

Function transforms (e.g. `jit`, `cond`, `grad`) use a `FunctionTransformNode` subclass — **not** `OperatorNode`. They are **only** registered in `patching.py` (not in `collection.py`). See `src/quantax/functional/jit.py` as reference.

### Function transformations (jit, cond, grad, etc.)

Each JAX function transformation is represented as a `FunctionTransformNode` subclass. These nodes maintain local `TraceData` and communicate with the global graph via equality no-op constraints on inputs/outputs, enabling nested transforms and cycle handling. Replayed recursively.

## Dependencies

- `jax` — core array computation
- `ortools` — MILP solver for scale optimization
- `rustworkx` — graph library for topological ordering of computation graph
- `pytreeclass` — PyTree-compatible dataclass base (`TreeClass`)
- `frozendict` — immutable dict for `Unit`
- `fastcore` — function copying utilities for patching
