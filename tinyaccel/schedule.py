"""Explicit loop and tiling schedules between graph IR and ISA lowering."""

from __future__ import annotations

from dataclasses import dataclass

from .ir import Graph, Operation


@dataclass(frozen=True)
class LoopSpec:
    """One scheduled loop axis."""

    axis: str
    extent: int
    tile: int
    kind: str = "spatial"

    def __post_init__(self) -> None:
        if self.extent <= 0 or self.tile <= 0:
            raise ValueError("loop extent and tile must be positive")
        if self.kind not in {"spatial", "reduction"}:
            raise ValueError(f"unsupported loop kind: {self.kind!r}")

    @property
    def tiles(self) -> int:
        return (self.extent + self.tile - 1) // self.tile


@dataclass(frozen=True)
class ScheduledOperation:
    """A graph operation annotated with an explicit loop nest."""

    operation: Operation
    loops: tuple[LoopSpec, ...]

    def __post_init__(self) -> None:
        loops = tuple(self.loops)
        axes = [loop.axis for loop in loops]
        if len(axes) != len(set(axes)):
            raise ValueError(
                f"operation {self.operation.op!r} has duplicate loop axes"
            )
        object.__setattr__(self, "loops", loops)

    def loop(self, axis: str) -> LoopSpec:
        for loop in self.loops:
            if loop.axis == axis:
                return loop
        raise KeyError(f"operation {self.operation.op!r} has no {axis!r} loop")

    @property
    def output_tile_shape(self) -> tuple[int, ...]:
        return tuple(loop.tile for loop in self.loops if loop.kind == "spatial")


@dataclass(frozen=True)
class Schedule:
    """The complete scheduled form of an optimized graph."""

    graph: Graph
    operations: tuple[ScheduledOperation, ...]

    def __post_init__(self) -> None:
        operations = tuple(self.operations)
        scheduled_operations = tuple(item.operation for item in operations)
        if scheduled_operations != self.graph.operations:
            raise ValueError(
                "schedule operations must match graph operations in program order"
            )
        object.__setattr__(self, "operations", operations)

    def __str__(self) -> str:
        lines = ["schedule {"]
        for scheduled in self.operations:
            operation = scheduled.operation
            output = f"%{operation.output.name}"
            if not scheduled.loops:
                lines.append(f"  {output} = {operation.op} [unscheduled]")
                continue
            loops = ", ".join(
                f"{loop.axis}:{loop.extent} tile={loop.tile} "
                f"{loop.kind} ({loop.tiles} tiles)"
                for loop in scheduled.loops
            )
            lines.append(f"  {output} = {operation.op} [{loops}]")
        lines.append("}")
        return "\n".join(lines)


def create_schedule(
    graph: Graph,
    *,
    tile_m: int = 32,
    tile_n: int = 32,
    tile_k: int = 32,
    tile_h: int = 8,
    tile_w: int = 8,
    tile_oc: int = 16,
) -> Schedule:
    """Create the deterministic default schedule for supported operations."""

    for name, value in (
        ("tile_m", tile_m),
        ("tile_n", tile_n),
        ("tile_k", tile_k),
        ("tile_h", tile_h),
        ("tile_w", tile_w),
        ("tile_oc", tile_oc),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")

    scheduled: list[ScheduledOperation] = []
    for operation in graph.operations:
        if operation.op == "constant":
            loops: tuple[LoopSpec, ...] = ()
        elif operation.op in {"matmul", "matmul_bias_relu"}:
            lhs, rhs = operation.inputs[:2]
            m_extent, k_extent = lhs.type.shape
            n_extent = rhs.type.shape[1]
            loops = (
                LoopSpec("m", m_extent, min(tile_m, m_extent)),
                LoopSpec("n", n_extent, min(tile_n, n_extent)),
                LoopSpec("k", k_extent, min(tile_k, k_extent), "reduction"),
            )
        elif operation.op in {"add", "relu"}:
            loops = _elementwise_loops(operation, tile_m, tile_n)
        elif operation.op == "conv2d":
            input_value, weight = operation.inputs
            n_size, output_h, output_w, output_c = operation.output.type.shape
            kernel_h, kernel_w, input_c, _ = weight.type.shape
            loops = (
                LoopSpec("n", n_size, 1),
                LoopSpec("h", output_h, min(tile_h, output_h)),
                LoopSpec("w", output_w, min(tile_w, output_w)),
                LoopSpec("oc", output_c, min(tile_oc, output_c)),
                LoopSpec("kh", kernel_h, kernel_h, "reduction"),
                LoopSpec("kw", kernel_w, kernel_w, "reduction"),
                LoopSpec("ic", input_c, input_c, "reduction"),
            )
        else:
            raise NotImplementedError(f"cannot schedule operation {operation.op!r}")
        scheduled.append(ScheduledOperation(operation, loops))
    return Schedule(graph, tuple(scheduled))


def _elementwise_loops(
    operation: Operation, tile_m: int, tile_n: int
) -> tuple[LoopSpec, ...]:
    shape = operation.output.type.shape
    loops: list[LoopSpec] = []
    for index, extent in enumerate(shape):
        tile = extent
        if index == len(shape) - 1:
            tile = min(tile_n, extent)
        elif index == len(shape) - 2:
            tile = min(tile_m, extent)
        loops.append(LoopSpec(f"d{index}", extent, tile))
    return tuple(loops)
