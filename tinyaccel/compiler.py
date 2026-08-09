"""Lower TinyAccel graph IR into the minimal accelerator ISA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .hardware import HardwareConfig
from .ir import Graph
from .isa import Instruction, Opcode, Program

if TYPE_CHECKING:
    from .simulator import SimulationReport


@dataclass(frozen=True)
class CompileOptions:
    """Tiling choices made by the first compiler implementation."""

    tile_m: int = 32
    tile_n: int = 32
    tile_k: int = 32

    def __post_init__(self) -> None:
        for name, value in (
            ("tile_m", self.tile_m),
            ("tile_n", self.tile_n),
            ("tile_k", self.tile_k),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")


class Executable:
    """A compiled program that can be run by the TinyAccel simulator."""

    def __init__(self, program: Program, hardware: HardwareConfig) -> None:
        self.program = program
        self.hardware = hardware
        self.last_report: SimulationReport | None = None

    def run(self, *inputs: np.ndarray, **named_inputs: np.ndarray) -> np.ndarray:
        from .simulator import Simulator

        if inputs and named_inputs:
            raise TypeError("use positional or named inputs, not both")
        input_names = tuple(self.program.input_types)
        if inputs:
            if len(inputs) != len(input_names):
                raise TypeError(f"expected {len(input_names)} inputs, got {len(inputs)}")
            feeds = dict(zip(input_names, inputs, strict=True))
        else:
            feeds = named_inputs

        result, report = Simulator(self.hardware).run(self.program, feeds)
        self.last_report = report
        return result

    def report(self) -> str:
        if self.last_report is None:
            raise RuntimeError("run the executable before requesting a report")
        return str(self.last_report)


def compile(
    graph: Graph,
    *,
    options: CompileOptions | None = None,
    hardware: HardwareConfig | None = None,
) -> Executable:
    """Compile a single MatMul graph into a tiled TinyAccel program."""

    options = options or CompileOptions()
    hardware = hardware or HardwareConfig()
    graph.validate()

    if len(graph.operations) != 1 or graph.operations[0].op != "matmul":
        raise NotImplementedError("v0.1 supports graphs containing exactly one matmul")
    if len(graph.outputs) != 1 or graph.outputs[0] != graph.operations[0].output:
        raise NotImplementedError("v0.1 requires the matmul result as the sole output")

    operation = graph.operations[0]
    lhs, rhs = operation.inputs
    output = operation.output
    if lhs not in graph.inputs or rhs not in graph.inputs or len(graph.inputs) != 2:
        raise NotImplementedError("v0.1 requires two graph inputs feeding matmul")
    if output.type.dtype != np.dtype("float32"):
        raise NotImplementedError("v0.1 simulator currently supports float32 only")

    m_size, k_size = lhs.type.shape
    _, n_size = rhs.type.shape
    max_m_extent = min(options.tile_m, m_size)
    max_n_extent = min(options.tile_n, n_size)
    max_k_extent = min(options.tile_k, k_size)
    itemsize = output.type.dtype.itemsize
    required_sram = (
        max_m_extent * max_k_extent
        + max_k_extent * max_n_extent
        + max_m_extent * max_n_extent
    ) * itemsize
    if required_sram > hardware.sram_bytes:
        raise ValueError(
            "tile configuration requires "
            f"{required_sram} SRAM bytes, but hardware has {hardware.sram_bytes}"
        )

    instructions: list[Instruction] = []

    for m_offset in range(0, m_size, options.tile_m):
        m_extent = min(options.tile_m, m_size - m_offset)
        for n_offset in range(0, n_size, options.tile_n):
            n_extent = min(options.tile_n, n_size - n_offset)
            instructions.append(
                Instruction(
                    Opcode.ZERO,
                    {"buffer": "acc", "shape": (m_extent, n_extent)},
                )
            )
            for k_offset in range(0, k_size, options.tile_k):
                k_extent = min(options.tile_k, k_size - k_offset)
                instructions.extend(
                    (
                        Instruction(
                            Opcode.DMA_LOAD,
                            {
                                "source": lhs.name,
                                "buffer": "lhs",
                                "offset": (m_offset, k_offset),
                                "shape": (m_extent, k_extent),
                            },
                        ),
                        Instruction(
                            Opcode.DMA_LOAD,
                            {
                                "source": rhs.name,
                                "buffer": "rhs",
                                "offset": (k_offset, n_offset),
                                "shape": (k_extent, n_extent),
                            },
                        ),
                        Instruction(
                            Opcode.MATMUL,
                            {"lhs": "lhs", "rhs": "rhs", "accumulator": "acc"},
                        ),
                    )
                )
            instructions.append(
                Instruction(
                    Opcode.DMA_STORE,
                    {
                        "buffer": "acc",
                        "output": output.name,
                        "offset": (m_offset, n_offset),
                        "shape": (m_extent, n_extent),
                    },
                )
            )

    input_types = {
        value.name: (value.type.shape, value.type.dtype) for value in graph.inputs
    }
    program = Program(
        tuple(instructions),
        input_types,
        output.name,
        output.type.shape,
        output.type.dtype,
    )
    return Executable(program, hardware)
