"""TinyAccel public API."""

from .compiler import CompileOptions, Executable, compile
from .hardware import HardwareConfig
from .ir import Graph, GraphBuilder, TensorType, parse_graph
from .reference import ReferenceExecutor, evaluate
from .passes import (
    AlgebraicSimplificationPass,
    ConstantFoldingPass,
    DeadCodeEliminationPass,
    MatmulBiasReluFusionPass,
    PassManager,
    PassResult,
    default_pipeline,
)
from .simulator import SimulationReport, TimelineEvent

__all__ = [
    "CompileOptions",
    "ConstantFoldingPass",
    "DeadCodeEliminationPass",
    "Executable",
    "Graph",
    "GraphBuilder",
    "HardwareConfig",
    "MatmulBiasReluFusionPass",
    "PassManager",
    "PassResult",
    "ReferenceExecutor",
    "SimulationReport",
    "TensorType",
    "TimelineEvent",
    "AlgebraicSimplificationPass",
    "compile",
    "evaluate",
    "default_pipeline",
    "parse_graph",
]
