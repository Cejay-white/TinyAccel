"""Functional execution and analytical performance model for TinyAccel."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import ceil
from typing import Mapping

import numpy as np

from .hardware import HardwareConfig
from .isa import Instruction, MemorySpace, Opcode, Program


@dataclass(frozen=True)
class TimelineEvent:
    instruction_index: int
    opcode: str
    start_cycle: int
    end_cycle: int

    @property
    def cycles(self) -> int:
        return self.end_cycle - self.start_cycle


@dataclass(frozen=True)
class SimulationReport:
    total_cycles: int
    instruction_counts: Mapping[str, int]
    cycles_by_opcode: Mapping[str, int]
    dram_bytes_read: int
    dram_bytes_written: int
    peak_sram_bytes: int
    timeline: tuple[TimelineEvent, ...]
    sram_bytes_read: int = 0
    sram_bytes_written: int = 0

    @property
    def layout_cycles(self) -> int:
        """Cycles spent materializing layout-changing local operations."""

        return sum(
            self.cycles_by_opcode.get(opcode.value, 0)
            for opcode in (Opcode.TRANSPOSE, Opcode.IM2COL, Opcode.RESHAPE)
        )

    def __str__(self) -> str:
        lines = ["TinyAccel Simulation Report", "=" * 28]
        for opcode in Opcode:
            name = opcode.value
            count = self.instruction_counts.get(name, 0)
            cycles = self.cycles_by_opcode.get(name, 0)
            if count:
                lines.append(f"{name:<10} instructions={count:<4} cycles={cycles}")
        lines.extend(
            (
                "-" * 28,
                f"Total cycles:       {self.total_cycles}",
                f"Layout cycles:      {self.layout_cycles}",
                f"DRAM bytes read:    {self.dram_bytes_read}",
                f"DRAM bytes written: {self.dram_bytes_written}",
                f"SRAM bytes read:    {self.sram_bytes_read}",
                f"SRAM bytes written: {self.sram_bytes_written}",
                f"Peak SRAM bytes:    {self.peak_sram_bytes}",
            )
        )
        return "\n".join(lines)

    def format_timeline(self, *, width: int = 48, max_events: int = 24) -> str:
        """Render a compact Gantt-style view of sequential instruction costs."""

        if width < 10:
            raise ValueError("timeline width must be at least 10")
        if max_events < 2:
            raise ValueError("max_events must be at least 2")

        events: list[TimelineEvent | None] = list(self.timeline)
        omitted = len(events) - max_events
        if omitted > 0:
            head_count = max_events // 2
            tail_count = max_events - head_count
            events = events[:head_count] + [None] + events[-tail_count:]

        lines = [
            "TinyAccel Instruction Timeline",
            f"cycles 0{' ' * max(1, width - len(str(self.total_cycles)) - 1)}"
            f"{self.total_cycles}",
        ]
        total = max(1, self.total_cycles)
        for event in events:
            if event is None:
                lines.append(f"     ... {omitted} instructions omitted ...")
                continue
            start = min(width - 1, event.start_cycle * width // total)
            end = min(
                width,
                max(start + 1, (event.end_cycle * width + total - 1) // total),
            )
            bar = " " * start + "#" * (end - start) + " " * (width - end)
            lines.append(
                f"{event.instruction_index:04d} {event.opcode:<9} "
                f"[{event.start_cycle:>6},{event.end_cycle:<6}) |{bar}|"
            )
        return "\n".join(lines)


class Simulator:
    """Execute ISA instructions and account for sequential resource costs."""

    def __init__(self, hardware: HardwareConfig) -> None:
        self.hardware = hardware

    def run(
        self, program: Program, feeds: Mapping[str, np.ndarray]
    ) -> tuple[np.ndarray, SimulationReport]:
        validated_feeds = self._validate_feeds(program, feeds)
        memory: dict[str, np.ndarray] = dict(validated_feeds)
        memory.update(
            {
                name: np.asarray(value).copy()
                for name, value in program.constants.items()
            }
        )
        value_types = dict(program.value_types)
        if not value_types:
            value_types[program.output_name] = (
                program.output_shape,
                program.output_dtype,
            )
        arena = None
        if program.memory_plan is not None:
            arena = np.empty(program.memory_plan.total_bytes, dtype=np.uint8)
            for name, allocation in program.memory_plan.allocations.items():
                shape, dtype = value_types[name]
                memory[name] = np.ndarray(
                    shape,
                    dtype=dtype,
                    buffer=arena,
                    offset=allocation.offset,
                )
        for name, (shape, dtype) in value_types.items():
            if name not in memory:
                memory[name] = np.empty(shape, dtype=dtype)
        buffers: dict[str, np.ndarray] = {}
        instruction_counts: Counter[str] = Counter()
        cycles_by_opcode: Counter[str] = Counter()
        dram_bytes_read = 0
        dram_bytes_written = 0
        sram_bytes_read = 0
        sram_bytes_written = 0
        planned_sram_bytes = 0 if arena is None else arena.nbytes
        peak_sram_bytes = planned_sram_bytes
        if peak_sram_bytes > self.hardware.sram_bytes:
            raise RuntimeError(
                f"memory plan uses {peak_sram_bytes} SRAM bytes, exceeding "
                f"the {self.hardware.sram_bytes}-byte capacity"
            )
        current_cycle = 0
        timeline: list[TimelineEvent] = []

        for index, instruction in enumerate(program.instructions):
            instruction_counts[instruction.opcode.value] += 1
            cycles, dram_read, dram_written, sram_read, sram_written = self._execute(
                instruction, memory, buffers
            )
            cycles_by_opcode[instruction.opcode.value] += cycles
            timeline.append(
                TimelineEvent(
                    index,
                    instruction.opcode.value,
                    current_cycle,
                    current_cycle + cycles,
                )
            )
            current_cycle += cycles
            dram_bytes_read += dram_read
            dram_bytes_written += dram_written
            sram_bytes_read += sram_read
            sram_bytes_written += sram_written
            current_sram = planned_sram_bytes + sum(
                buffer.nbytes for buffer in buffers.values()
            )
            peak_sram_bytes = max(peak_sram_bytes, current_sram)
            if current_sram > self.hardware.sram_bytes:
                raise RuntimeError(
                    f"program used {current_sram} SRAM bytes, exceeding "
                    f"the {self.hardware.sram_bytes}-byte capacity"
                )

        report = SimulationReport(
            total_cycles=sum(cycles_by_opcode.values()),
            instruction_counts=dict(instruction_counts),
            cycles_by_opcode=dict(cycles_by_opcode),
            dram_bytes_read=dram_bytes_read,
            dram_bytes_written=dram_bytes_written,
            peak_sram_bytes=peak_sram_bytes,
            timeline=tuple(timeline),
            sram_bytes_read=sram_bytes_read,
            sram_bytes_written=sram_bytes_written,
        )
        return memory[program.output_name], report

    def _execute(
        self,
        instruction: Instruction,
        memory: dict[str, np.ndarray],
        buffers: dict[str, np.ndarray],
    ) -> tuple[int, int, int, int, int]:
        operands = instruction.operands

        if instruction.opcode is Opcode.ZERO:
            buffer = np.zeros(operands["shape"], dtype=np.float32)
            buffers[operands["buffer"]] = buffer
            cycles = ceil(buffer.size / self.hardware.macs_per_cycle)
            return cycles, 0, 0, 0, 0

        if instruction.opcode is Opcode.DMA_LOAD:
            source = memory[operands["source"]]
            if operands.get("padded", False):
                tile, transferred_bytes = _read_padded_tile(
                    source, operands["offset"], operands["shape"]
                )
            else:
                tile = _read_tile(source, operands["offset"], operands["shape"])
                transferred_bytes = tile.nbytes
            buffers[operands["buffer"]] = tile
            cycles = ceil(transferred_bytes / self.hardware.dma_bytes_per_cycle)
            source_space = MemorySpace(operands["space"])
            dram_read = transferred_bytes if source_space is MemorySpace.DRAM else 0
            sram_read = transferred_bytes if source_space is MemorySpace.SRAM else 0
            return cycles, dram_read, 0, sram_read, tile.nbytes

        if instruction.opcode is Opcode.IM2COL:
            source = buffers[operands["input"]]
            result = _im2col(
                source,
                kernel=operands["kernel"],
                stride=operands["stride"],
                dilation=operands["dilation"],
                output_shape=operands["output_shape"],
            )
            buffers[operands["output"]] = result
            cycles = ceil(
                result.size / self.hardware.vector_elements_per_cycle
            )
            return cycles, 0, 0, 0, 0

        if instruction.opcode is Opcode.RESHAPE:
            source = buffers[operands["input"]]
            buffers[operands["output"]] = source.reshape(operands["shape"])
            return 0, 0, 0, 0, 0

        if instruction.opcode is Opcode.MATMUL:
            lhs = buffers[operands["lhs"]]
            rhs = buffers[operands["rhs"]]
            accumulator = buffers[operands["accumulator"]]
            accumulator += lhs @ rhs
            macs = lhs.shape[0] * rhs.shape[1] * lhs.shape[1]
            return ceil(macs / self.hardware.macs_per_cycle), 0, 0, 0, 0

        if instruction.opcode is Opcode.ADD:
            result = buffers[operands["lhs"]] + buffers[operands["rhs"]]
            buffers[operands["output"]] = np.asarray(result, dtype=np.float32)
            cycles = ceil(result.size / self.hardware.macs_per_cycle)
            return cycles, 0, 0, 0, 0

        if instruction.opcode is Opcode.RELU:
            source = buffers[operands["input"]]
            result = np.maximum(source, 0)
            buffers[operands["output"]] = result
            cycles = ceil(result.size / self.hardware.macs_per_cycle)
            return cycles, 0, 0, 0, 0

        if instruction.opcode is Opcode.TRANSPOSE:
            source = buffers[operands["input"]]
            result = np.transpose(source, operands["permutation"]).copy()
            buffers[operands["output"]] = result
            cycles = ceil(
                result.size / self.hardware.vector_elements_per_cycle
            )
            return cycles, 0, 0, 0, 0

        if instruction.opcode is Opcode.CONV2D:
            input_tile = buffers[operands["input"]]
            weight = buffers[operands["weight"]]
            accumulator = buffers[operands["accumulator"]]
            stride_h, stride_w = operands["stride"]
            dilation_h, dilation_w = operands["dilation"]
            kernel_h, kernel_w, input_c, _ = weight.shape
            _, output_h, output_w, output_c = accumulator.shape
            effective_h = (kernel_h - 1) * dilation_h + 1
            effective_w = (kernel_w - 1) * dilation_w + 1
            for output_y in range(output_h):
                input_y = output_y * stride_h
                for output_x in range(output_w):
                    input_x = output_x * stride_w
                    patch = input_tile[
                        0,
                        input_y : input_y + effective_h : dilation_h,
                        input_x : input_x + effective_w : dilation_w,
                        :,
                    ]
                    accumulator[0, output_y, output_x, :] += np.tensordot(
                        patch, weight, axes=((0, 1, 2), (0, 1, 2))
                    )
            macs = output_h * output_w * output_c * kernel_h * kernel_w * input_c
            return ceil(macs / self.hardware.macs_per_cycle), 0, 0, 0, 0

        if instruction.opcode is Opcode.DMA_STORE:
            tile = buffers[operands["buffer"]]
            _write_tile(
                memory[operands["output"]],
                operands["offset"],
                operands["shape"],
                tile,
            )
            cycles = ceil(tile.nbytes / self.hardware.dma_bytes_per_cycle)
            buffers.clear()
            destination_space = MemorySpace(operands["space"])
            dram_written = (
                tile.nbytes if destination_space is MemorySpace.DRAM else 0
            )
            sram_written = (
                tile.nbytes if destination_space is MemorySpace.SRAM else 0
            )
            return cycles, 0, dram_written, tile.nbytes, sram_written

        raise ValueError(f"unsupported opcode: {instruction.opcode}")

    @staticmethod
    def _validate_feeds(
        program: Program, feeds: Mapping[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        expected_names = set(program.input_types)
        actual_names = set(feeds)
        if actual_names != expected_names:
            missing = sorted(expected_names - actual_names)
            extra = sorted(actual_names - expected_names)
            raise ValueError(f"input name mismatch; missing={missing}, extra={extra}")

        validated: dict[str, np.ndarray] = {}
        for name, (expected_shape, expected_dtype) in program.input_types.items():
            value = np.asarray(feeds[name])
            if value.shape != expected_shape:
                raise ValueError(
                    f"input {name!r} has shape {value.shape}, expected {expected_shape}"
                )
            if value.dtype != expected_dtype:
                raise ValueError(
                    f"input {name!r} has dtype {value.dtype}, expected {expected_dtype}"
                )
            validated[name] = value
        return validated


def _read_tile(
    source: np.ndarray,
    offset: tuple[int, ...],
    shape: tuple[int, ...],
) -> np.ndarray:
    if not shape:
        return source.copy()
    slices = tuple(
        slice(start, start + extent) for start, extent in zip(offset, shape)
    )
    return source[slices].copy()


def _im2col(
    input_tile: np.ndarray,
    *,
    kernel: tuple[int, int],
    stride: tuple[int, int],
    dilation: tuple[int, int],
    output_shape: tuple[int, int],
) -> np.ndarray:
    """Materialize one NHWC input tile as a two-dimensional patch matrix."""

    if input_tile.ndim != 4 or input_tile.shape[0] != 1:
        raise ValueError("IM2COL requires an NHWC tile with batch extent one")
    kernel_h, kernel_w = kernel
    stride_h, stride_w = stride
    dilation_h, dilation_w = dilation
    output_h, output_w = output_shape
    input_c = input_tile.shape[3]
    columns = np.empty(
        (output_h * output_w, kernel_h * kernel_w * input_c),
        dtype=input_tile.dtype,
    )
    row = 0
    effective_h = (kernel_h - 1) * dilation_h + 1
    effective_w = (kernel_w - 1) * dilation_w + 1
    for output_y in range(output_h):
        input_y = output_y * stride_h
        for output_x in range(output_w):
            input_x = output_x * stride_w
            patch = input_tile[
                0,
                input_y : input_y + effective_h : dilation_h,
                input_x : input_x + effective_w : dilation_w,
                :,
            ]
            if patch.shape != (kernel_h, kernel_w, input_c):
                raise ValueError("IM2COL input tile does not cover the output patch")
            columns[row, :] = patch.reshape(-1)
            row += 1
    return columns


def _read_padded_tile(
    source: np.ndarray,
    offset: tuple[int, ...],
    shape: tuple[int, ...],
) -> tuple[np.ndarray, int]:
    """Read a possibly out-of-bounds tile, filling the halo with zeros."""

    tile = np.zeros(shape, dtype=source.dtype)
    source_slices: list[slice] = []
    tile_slices: list[slice] = []
    for dimension, start, extent in zip(source.shape, offset, shape):
        source_start = max(0, start)
        source_stop = min(dimension, start + extent)
        if source_stop <= source_start:
            return tile, 0
        tile_start = source_start - start
        source_slices.append(slice(source_start, source_stop))
        tile_slices.append(slice(tile_start, tile_start + source_stop - source_start))
    tile[tuple(tile_slices)] = source[tuple(source_slices)]
    transferred_bytes = int(source[tuple(source_slices)].nbytes)
    return tile, transferred_bytes


def _write_tile(
    target: np.ndarray,
    offset: tuple[int, ...],
    shape: tuple[int, ...],
    tile: np.ndarray,
) -> None:
    if not shape:
        target[...] = tile
        return
    slices = tuple(
        slice(start, start + extent) for start, extent in zip(offset, shape)
    )
    target[slices] = tile
