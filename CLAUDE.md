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

### Adding a new operation

Each operation needs:
1. An implementation function in `src/quantax/functional/numpy/` (or similar) that handles both eager `Unitful` and tracer `UnitfulTracer` cases.
2. A `constraints_*` function that returns OR-Tools `BoundedLinearExpression` constraints relating input and output scale variables.
3. Registration in `src/quantax/functional/collection.py` in both `CONSTRAINTS_DICT` and `FUNCTION_DICT`.

### Function transformations (jit, cond, grad, etc.)

Each JAX function transformation is represented as a `FunctionTransformNode` subclass. These nodes maintain local `TraceData` (not added to the global graph directly) and communicate with the global graph via equality no-op constraints on inputs/outputs. This enables handling cycles (e.g. `while` loops). Nested transforms build a tree of nodes that are replayed recursively.

## Dependencies

- `jax` — core array computation
- `ortools` — MILP solver for scale optimization
- `rustworkx` — graph library for topological ordering of computation graph
- `pytreeclass` — PyTree-compatible dataclass base (`TreeClass`)
- `frozendict` — immutable dict for `Unit`
- `fastcore` — function copying utilities for patching
