"""TinyAccel public API."""

from .compiler import CompileOptions, Executable, compile
from .hardware import HardwareConfig
from .isa import MemorySpace
from .ir import Graph, GraphBuilder, TensorType, layout_permutation, parse_graph
from .memory import (
    BufferAllocation,
    MemoryPlan,
    ValueLifetime,
    analyze_lifetimes,
    plan_memory,
)
from .reference import ReferenceExecutor, conv2d_nhwc, evaluate
from .passes import (
    AlgebraicSimplificationPass,
    CanonicalizeConv2dLayoutsPass,
    ConstantFoldingPass,
    DeadCodeEliminationPass,
    LayoutTransformSimplificationPass,
    MatmulBiasReluFusionPass,
    PassManager,
    PassResult,
    default_pipeline,
)
from .simulator import ExecutionResource, SimulationReport, TimelineEvent
from .schedule import LoopSpec, Schedule, ScheduledOperation, create_schedule

__all__ = [
    "CompileOptions",
    "BufferAllocation",
    "CanonicalizeConv2dLayoutsPass",
    "ConstantFoldingPass",
    "DeadCodeEliminationPass",
    "Executable",
    "ExecutionResource",
    "Graph",
    "GraphBuilder",
    "HardwareConfig",
    "MatmulBiasReluFusionPass",
    "MemorySpace",
    "MemoryPlan",
    "PassManager",
    "PassResult",
    "ReferenceExecutor",
    "SimulationReport",
    "LoopSpec",
    "LayoutTransformSimplificationPass",
    "Schedule",
    "ScheduledOperation",
    "TensorType",
    "TimelineEvent",
    "ValueLifetime",
    "AlgebraicSimplificationPass",
    "compile",
    "conv2d_nhwc",
    "create_schedule",
    "evaluate",
    "layout_permutation",
    "analyze_lifetimes",
    "default_pipeline",
    "parse_graph",
    "plan_memory",
]
