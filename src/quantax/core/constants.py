"""This determines the maximum size where an array can be saved statically for optimization during jit tracing"""

MAX_STATIC_OPTIMIZED_SIZE: int = int(1e5)

"""
Relative tolerance for floating point comparisons
TODO: 
    adjust RTOL_COMPARISON and ATOL_COMPARISON dynamically
    based on the numeric precision (e.g., float16, float32, float64).
    For now, these fixed values are sufficient for most use cases.

| Precision Type | RTOL (Relative Tolerance) | ATOL (Absolute Tolerance) |
| -------------- | ------------------------- | ------------------------- |
| `float16`      | 1e-2 - 1e-3               | 1e-3 - 1e-4               |
| `float32`      | 1e-4 - 1e-5               | 1e-6 - 1e-7               |
| `float64`      | 1e-6 - 1e-8               | 1e-9 - 1e-12              | 
"""
RTOL_COMPARSION = 1e-5
ATOL_COMPARSION = 1e-7
