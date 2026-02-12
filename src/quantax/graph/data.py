from dataclasses import dataclass
from typing import Any

from rustworkx import PyDiGraph


@dataclass(frozen=True, kw_only=True)
class GraphData:
    graph: PyDiGraph
    args: Any
    kwargs: Any
    trace_args: Any
    trace_kwargs: Any
    trace_output: Any
