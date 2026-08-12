"""TinyAccel public API."""

from .compiler import CompileOptions, Executable, compile
from .hardware import HardwareConfig
from .ir import Graph, GraphBuilder, TensorType, parse_graph
from .reference import ReferenceExecutor, evaluate
from .memory import (
    BufferAllocation,
    MemoryPlan,
    ValueLifetime,
    analyze_lifetimes,
    plan_memory,
)
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
from .schedule import LoopSpec, Schedule, ScheduledOperation, create_schedule

__all__ = [
    "CompileOptions",
    "BufferAllocation",
    "ConstantFoldingPass",
    "DeadCodeEliminationPass",
    "Executable",
    "Graph",
    "GraphBuilder",
    "HardwareConfig",
    "MatmulBiasReluFusionPass",
    "MemoryPlan",
    "PassManager",
    "PassResult",
    "ReferenceExecutor",
    "SimulationReport",
    "LoopSpec",
    "Schedule",
    "ScheduledOperation",
    "TensorType",
    "TimelineEvent",
    "ValueLifetime",
    "AlgebraicSimplificationPass",
    "compile",
    "create_schedule",
    "evaluate",
    "analyze_lifetimes",
    "default_pipeline",
    "parse_graph",
    "plan_memory",
]
