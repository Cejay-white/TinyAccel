"""Functional execution and analytical performance model for TinyAccel."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import ceil
from typing import Mapping

import numpy as np

from .hardware import HardwareConfig
from .isa import Instruction, Opcode, Program


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
                f"DRAM bytes read:    {self.dram_bytes_read}",
                f"DRAM bytes written: {self.dram_bytes_written}",
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
        output = np.empty(program.output_shape, dtype=program.output_dtype)
        buffers: dict[str, np.ndarray] = {}
        instruction_counts: Counter[str] = Counter()
        cycles_by_opcode: Counter[str] = Counter()
        bytes_read = 0
        bytes_written = 0
        peak_sram_bytes = 0
        current_cycle = 0
        timeline: list[TimelineEvent] = []

        for index, instruction in enumerate(program.instructions):
            instruction_counts[instruction.opcode.value] += 1
            cycles, read, written = self._execute(
                instruction, validated_feeds, output, buffers
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
            bytes_read += read
            bytes_written += written
            current_sram = sum(buffer.nbytes for buffer in buffers.values())
            peak_sram_bytes = max(peak_sram_bytes, current_sram)
            if current_sram > self.hardware.sram_bytes:
                raise RuntimeError(
                    f"program used {current_sram} SRAM bytes, exceeding "
                    f"the {self.hardware.sram_bytes}-byte capacity"
                )

        report = SimulationReport(
            sum(cycles_by_opcode.values()),
            dict(instruction_counts),
            dict(cycles_by_opcode),
            bytes_read,
            bytes_written,
            peak_sram_bytes,
            tuple(timeline),
        )
        return output, report

    def _execute(
        self,
        instruction: Instruction,
        feeds: Mapping[str, np.ndarray],
        output: np.ndarray,
        buffers: dict[str, np.ndarray],
    ) -> tuple[int, int, int]:
        operands = instruction.operands

        if instruction.opcode is Opcode.ZERO:
            buffer = np.zeros(operands["shape"], dtype=output.dtype)
            buffers[operands["buffer"]] = buffer
            cycles = ceil(buffer.size / self.hardware.macs_per_cycle)
            return cycles, 0, 0

        if instruction.opcode is Opcode.DMA_LOAD:
            source = feeds[operands["source"]]
            row, column = operands["offset"]
            rows, columns = operands["shape"]
            tile = source[row : row + rows, column : column + columns].copy()
            buffers[operands["buffer"]] = tile
            cycles = ceil(tile.nbytes / self.hardware.dma_bytes_per_cycle)
            return cycles, tile.nbytes, 0

        if instruction.opcode is Opcode.MATMUL:
            lhs = buffers[operands["lhs"]]
            rhs = buffers[operands["rhs"]]
            accumulator = buffers[operands["accumulator"]]
            accumulator += lhs @ rhs
            macs = lhs.shape[0] * rhs.shape[1] * lhs.shape[1]
            return ceil(macs / self.hardware.macs_per_cycle), 0, 0

        if instruction.opcode is Opcode.DMA_STORE:
            tile = buffers[operands["buffer"]]
            row, column = operands["offset"]
            rows, columns = operands["shape"]
            output[row : row + rows, column : column + columns] = tile
            cycles = ceil(tile.nbytes / self.hardware.dma_bytes_per_cycle)
            return cycles, 0, tile.nbytes

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
