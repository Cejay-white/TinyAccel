"""Tensor lifetime analysis and linear-scan SRAM memory planning."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from .ir import Graph, Value


@dataclass(frozen=True)
class ValueLifetime:
    """Inclusive operation-index interval in which an SSA value is live."""

    value: Value
    start: int
    end: int

    def overlaps(self, other: "ValueLifetime") -> bool:
        return self.start <= other.end and other.start <= self.end


@dataclass(frozen=True)
class BufferAllocation:
    """A byte range assigned to one materialized tensor value."""

    value: Value
    offset: int
    size: int
    lifetime: ValueLifetime

    @property
    def end_offset(self) -> int:
        return self.offset + self.size


@dataclass(frozen=True)
class MemoryPlan:
    """Reusable SRAM arena allocation for operation results."""

    allocations: Mapping[str, BufferAllocation]
    total_bytes: int
    alignment: int
    capacity_bytes: int | None = None

    def __post_init__(self) -> None:
        allocations = dict(self.allocations)
        if self.total_bytes < 0:
            raise ValueError("memory plan total_bytes cannot be negative")
        if self.alignment <= 0:
            raise ValueError("memory plan alignment must be positive")
        if self.capacity_bytes is not None and self.capacity_bytes <= 0:
            raise ValueError("memory plan capacity_bytes must be positive")
        for allocation in allocations.values():
            required = (
                allocation.value.type.dtype.itemsize
                * _element_count(allocation.value.type.shape)
            )
            if allocation.offset < 0 or allocation.offset % self.alignment:
                raise ValueError(
                    f"allocation for %{allocation.value.name} has unaligned offset "
                    f"{allocation.offset}"
                )
            if allocation.size < required or allocation.size % self.alignment:
                raise ValueError(
                    f"allocation for %{allocation.value.name} has invalid size "
                    f"{allocation.size}"
                )
            if allocation.end_offset > self.total_bytes:
                raise ValueError(
                    f"allocation for %{allocation.value.name} exceeds SRAM arena"
                )
        planned = tuple(allocations.values())
        for index, lhs in enumerate(planned):
            for rhs in planned[index + 1 :]:
                address_overlap = (
                    lhs.offset < rhs.end_offset and rhs.offset < lhs.end_offset
                )
                if address_overlap and lhs.lifetime.overlaps(rhs.lifetime):
                    raise ValueError(
                        f"live allocations %{lhs.value.name} and %{rhs.value.name} "
                        "overlap in SRAM"
                    )
        if self.capacity_bytes is not None and self.total_bytes > self.capacity_bytes:
            raise ValueError(
                f"memory plan requires {self.total_bytes} SRAM bytes, but capacity is "
                f"{self.capacity_bytes}"
            )
        object.__setattr__(self, "allocations", MappingProxyType(allocations))

    def allocation(self, value: Value | str) -> BufferAllocation:
        name = value if isinstance(value, str) else value.name
        try:
            return self.allocations[name]
        except KeyError as error:
            raise KeyError(f"value %{name} has no planned buffer") from error

    def __str__(self) -> str:
        lines = [
            "TinyAccel Memory Plan",
            f"sram_arena_bytes={self.total_bytes} alignment={self.alignment}",
        ]
        for allocation in self.allocations.values():
            lifetime = allocation.lifetime
            lines.append(
                f"%{allocation.value.name:<12} offset={allocation.offset:<8} "
                f"size={allocation.size:<8} live=[{lifetime.start},{lifetime.end}]"
            )
        return "\n".join(lines)


def analyze_lifetimes(graph: Graph) -> tuple[ValueLifetime, ...]:
    """Compute deterministic live intervals for every graph value."""

    operation_index = {
        id(operation): index for index, operation in enumerate(graph.operations)
    }
    graph_end = len(graph.operations)
    output_values = set(graph.outputs)
    lifetimes: list[ValueLifetime] = []

    for value in graph.values:
        producer = graph.producer(value)
        start = -1 if producer is None else operation_index[id(producer)]
        uses = [operation_index[id(user)] for user in graph.users(value)]
        end = max(uses, default=start)
        if value in output_values:
            end = graph_end
        lifetimes.append(ValueLifetime(value, start, end))
    return tuple(lifetimes)


def plan_memory(
    graph: Graph,
    *,
    alignment: int = 64,
    capacity_bytes: int | None = None,
    materialize_outputs: bool = False,
) -> MemoryPlan:
    """Allocate intermediate results in SRAM with first-fit lifetime reuse.

    Graph outputs remain external by default so compiled programs can DMA their
    final tiles directly to DRAM. Set ``materialize_outputs`` to retain outputs
    in the SRAM arena for standalone memory-planning experiments.
    """

    if alignment <= 0:
        raise ValueError(f"alignment must be positive, got {alignment}")
    if capacity_bytes is not None and capacity_bytes <= 0:
        raise ValueError(f"capacity_bytes must be positive, got {capacity_bytes}")

    lifetimes = {item.value: item for item in analyze_lifetimes(graph)}
    constants = {
        operation.output for operation in graph.operations if operation.op == "constant"
    }
    external_outputs = set() if materialize_outputs else set(graph.outputs)
    materialized = [
        operation.output
        for operation in graph.operations
        if operation.output not in constants and operation.output not in external_outputs
    ]
    allocations: dict[str, BufferAllocation] = {}
    active: list[BufferAllocation] = []
    free_blocks: list[tuple[int, int]] = []
    arena_end = 0

    for value in materialized:
        lifetime = lifetimes[value]
        still_active: list[BufferAllocation] = []
        for allocation in active:
            if allocation.lifetime.end < lifetime.start:
                free_blocks.append((allocation.offset, allocation.size))
            else:
                still_active.append(allocation)
        active = still_active
        free_blocks = _coalesce(free_blocks)

        size = _align(
            value.type.dtype.itemsize * _element_count(value.type.shape), alignment
        )
        offset = _take_first_fit(free_blocks, size)
        if offset is None:
            offset = _align(arena_end, alignment)
            arena_end = offset + size
            if capacity_bytes is not None and arena_end > capacity_bytes:
                raise ValueError(
                    f"memory plan requires {arena_end} SRAM bytes, but hardware has "
                    f"{capacity_bytes}"
                )
        allocation = BufferAllocation(value, offset, size, lifetime)
        allocations[value.name] = allocation
        active.append(allocation)

    return MemoryPlan(allocations, arena_end, alignment, capacity_bytes)


def _take_first_fit(blocks: list[tuple[int, int]], size: int) -> int | None:
    for index, (offset, available) in enumerate(blocks):
        if available < size:
            continue
        if available == size:
            blocks.pop(index)
        else:
            blocks[index] = (offset + size, available - size)
        return offset
    return None


def _coalesce(blocks: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for offset, size in sorted(blocks):
        if merged and merged[-1][0] + merged[-1][1] == offset:
            previous_offset, previous_size = merged[-1]
            merged[-1] = (previous_offset, previous_size + size)
        else:
            merged.append((offset, size))
    return merged


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _element_count(shape: tuple[int, ...]) -> int:
    count = 1
    for extent in shape:
        count *= extent
    return count
