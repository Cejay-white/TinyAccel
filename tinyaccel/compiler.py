"""Lower TinyAccel graph IR into the minimal accelerator ISA."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING, Iterable

import numpy as np

from .hardware import HardwareConfig
from .ir import Graph, Operation, TensorType
from .isa import Instruction, Opcode, Program
from .passes import default_pipeline

if TYPE_CHECKING:
    from .simulator import SimulationReport


@dataclass(frozen=True)
class CompileOptions:
    """Optimization and tiling choices made by the compiler."""

    tile_m: int = 32
    tile_n: int = 32
    tile_k: int = 32
    optimize: bool = True

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

    def __init__(
        self,
        program: Program,
        hardware: HardwareConfig,
        graph: Graph,
    ) -> None:
        self.program = program
        self.hardware = hardware
        self.graph = graph
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

    def timeline(self, *, width: int = 48, max_events: int = 24) -> str:
        if self.last_report is None:
            raise RuntimeError("run the executable before requesting a timeline")
        return self.last_report.format_timeline(width=width, max_events=max_events)


def compile(
    graph: Graph,
    *,
    options: CompileOptions | None = None,
    hardware: HardwareConfig | None = None,
) -> Executable:
    """Compile a supported graph into a tiled TinyAccel program."""

    options = options or CompileOptions()
    hardware = hardware or HardwareConfig()
    graph.validate()
    lowered_graph = default_pipeline().run(graph) if options.optimize else graph

    if len(lowered_graph.outputs) != 1:
        raise NotImplementedError("accelerator backend currently requires one output")
    for value in lowered_graph.values:
        if value.type.dtype != np.dtype("float32"):
            raise NotImplementedError("accelerator backend currently supports float32")

    instructions: list[Instruction] = []
    constants: dict[str, np.ndarray] = {}
    for operation in lowered_graph.operations:
        if operation.op == "constant":
            constants[operation.output.name] = np.asarray(
                operation.attributes["value"], dtype=operation.output.type.dtype
            ).copy()
        elif operation.op == "matmul":
            _lower_matmul(operation, instructions, options, hardware)
        elif operation.op == "add":
            _lower_add(operation, instructions, options, hardware)
        elif operation.op == "relu":
            _lower_relu(operation, instructions, options, hardware)
        elif operation.op == "matmul_bias_relu":
            _lower_matmul_bias_relu(operation, instructions, options, hardware)
        else:
            raise NotImplementedError(
                f"accelerator backend does not support {operation.op!r}"
            )

    input_types = {
        value.name: (value.type.shape, value.type.dtype)
        for value in lowered_graph.inputs
    }
    value_types = {
        value.name: (value.type.shape, value.type.dtype)
        for value in lowered_graph.values
    }
    output = lowered_graph.outputs[0]
    program = Program(
        tuple(instructions),
        input_types,
        output.name,
        output.type.shape,
        output.type.dtype,
        value_types,
        constants,
    )
    return Executable(program, hardware, lowered_graph)


def _lower_matmul(
    operation: Operation,
    instructions: list[Instruction],
    options: CompileOptions,
    hardware: HardwareConfig,
) -> None:
    lhs, rhs = operation.inputs
    output = operation.output
    m_size, k_size = lhs.type.shape
    _, n_size = rhs.type.shape

    for m_offset in range(0, m_size, options.tile_m):
        m_extent = min(options.tile_m, m_size - m_offset)
        for n_offset in range(0, n_size, options.tile_n):
            n_extent = min(options.tile_n, n_size - n_offset)
            max_k_extent = min(options.tile_k, k_size)
            _require_sram(
                (
                    m_extent * max_k_extent
                    + max_k_extent * n_extent
                    + m_extent * n_extent
                )
                * output.type.dtype.itemsize,
                hardware,
                "matmul tile",
            )
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
                        _load(
                            lhs.name,
                            "lhs",
                            (m_offset, k_offset),
                            (m_extent, k_extent),
                        ),
                        _load(
                            rhs.name,
                            "rhs",
                            (k_offset, n_offset),
                            (k_extent, n_extent),
                        ),
                        Instruction(
                            Opcode.MATMUL,
                            {"lhs": "lhs", "rhs": "rhs", "accumulator": "acc"},
                        ),
                    )
                )
            instructions.append(
                _store(
                    "acc",
                    output.name,
                    (m_offset, n_offset),
                    (m_extent, n_extent),
                )
            )


def _lower_add(
    operation: Operation,
    instructions: list[Instruction],
    options: CompileOptions,
    hardware: HardwareConfig,
) -> None:
    lhs, rhs = operation.inputs
    output = operation.output
    for offset, shape in _output_tiles(output.type, options):
        lhs_offset, lhs_shape = _broadcast_slice(lhs.type.shape, output.type.shape, offset, shape)
        rhs_offset, rhs_shape = _broadcast_slice(rhs.type.shape, output.type.shape, offset, shape)
        required = (
            _element_count(lhs_shape)
            + _element_count(rhs_shape)
            + _element_count(shape)
        ) * output.type.dtype.itemsize
        _require_sram(required, hardware, "add tile")
        instructions.extend(
            (
                _load(lhs.name, "lhs", lhs_offset, lhs_shape),
                _load(rhs.name, "rhs", rhs_offset, rhs_shape),
                Instruction(
                    Opcode.ADD,
                    {"lhs": "lhs", "rhs": "rhs", "output": "acc"},
                ),
                _store("acc", output.name, offset, shape),
            )
        )


def _lower_relu(
    operation: Operation,
    instructions: list[Instruction],
    options: CompileOptions,
    hardware: HardwareConfig,
) -> None:
    source = operation.inputs[0]
    output = operation.output
    for offset, shape in _output_tiles(output.type, options):
        required = 2 * _element_count(shape) * output.type.dtype.itemsize
        _require_sram(required, hardware, "relu tile")
        instructions.extend(
            (
                _load(source.name, "lhs", offset, shape),
                Instruction(Opcode.RELU, {"input": "lhs", "output": "acc"}),
                _store("acc", output.name, offset, shape),
            )
        )


def _lower_matmul_bias_relu(
    operation: Operation,
    instructions: list[Instruction],
    options: CompileOptions,
    hardware: HardwareConfig,
) -> None:
    lhs, rhs, bias = operation.inputs
    output = operation.output
    m_size, k_size = lhs.type.shape
    _, n_size = rhs.type.shape

    for m_offset in range(0, m_size, options.tile_m):
        m_extent = min(options.tile_m, m_size - m_offset)
        for n_offset in range(0, n_size, options.tile_n):
            n_extent = min(options.tile_n, n_size - n_offset)
            max_k_extent = min(options.tile_k, k_size)
            bias_offset, bias_shape = _broadcast_slice(
                bias.type.shape,
                output.type.shape,
                (m_offset, n_offset),
                (m_extent, n_extent),
            )
            required = max(
                (
                    m_extent * max_k_extent
                    + max_k_extent * n_extent
                    + m_extent * n_extent
                )
                * output.type.dtype.itemsize,
                (
                    m_extent * max_k_extent
                    + _element_count(bias_shape)
                    + m_extent * n_extent
                )
                * output.type.dtype.itemsize,
            )
            _require_sram(required, hardware, "fused matmul tile")
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
                        _load(
                            lhs.name,
                            "lhs",
                            (m_offset, k_offset),
                            (m_extent, k_extent),
                        ),
                        _load(
                            rhs.name,
                            "rhs",
                            (k_offset, n_offset),
                            (k_extent, n_extent),
                        ),
                        Instruction(
                            Opcode.MATMUL,
                            {"lhs": "lhs", "rhs": "rhs", "accumulator": "acc"},
                        ),
                    )
                )
            instructions.extend(
                (
                    _load(bias.name, "rhs", bias_offset, bias_shape),
                    Instruction(
                        Opcode.ADD,
                        {"lhs": "acc", "rhs": "rhs", "output": "acc"},
                    ),
                    Instruction(Opcode.RELU, {"input": "acc", "output": "acc"}),
                    _store(
                        "acc",
                        output.name,
                        (m_offset, n_offset),
                        (m_extent, n_extent),
                    ),
                )
            )


def _output_tiles(
    tensor_type: TensorType, options: CompileOptions
) -> Iterable[tuple[tuple[int, ...], tuple[int, ...]]]:
    shape = tensor_type.shape
    if not shape:
        yield (), ()
        return
    tile_shape = list(shape)
    tile_shape[-1] = min(shape[-1], options.tile_n)
    if len(shape) >= 2:
        tile_shape[-2] = min(shape[-2], options.tile_m)
    ranges = [range(0, extent, tile) for extent, tile in zip(shape, tile_shape)]
    for offset in product(*ranges):
        extent = tuple(
            min(tile, dimension - start)
            for start, tile, dimension in zip(offset, tile_shape, shape)
        )
        yield tuple(offset), extent


def _broadcast_slice(
    source_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
    output_offset: tuple[int, ...],
    output_extent: tuple[int, ...],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    padding = len(output_shape) - len(source_shape)
    padded_shape = (1,) * padding + source_shape
    offsets: list[int] = []
    extents: list[int] = []
    for dimension, offset, extent in zip(padded_shape, output_offset, output_extent):
        offsets.append(0 if dimension == 1 else offset)
        extents.append(1 if dimension == 1 else extent)
    if padding:
        offsets = offsets[padding:]
        extents = extents[padding:]
    return tuple(offsets), tuple(extents)


def _load(
    source: str,
    buffer: str,
    offset: tuple[int, ...],
    shape: tuple[int, ...],
) -> Instruction:
    return Instruction(
        Opcode.DMA_LOAD,
        {"source": source, "buffer": buffer, "offset": offset, "shape": shape},
    )


def _store(
    buffer: str,
    output: str,
    offset: tuple[int, ...],
    shape: tuple[int, ...],
) -> Instruction:
    return Instruction(
        Opcode.DMA_STORE,
        {"buffer": buffer, "output": output, "offset": offset, "shape": shape},
    )


def _element_count(shape: tuple[int, ...]) -> int:
    return int(np.prod(shape, dtype=np.int64)) if shape else 1


def _require_sram(
    required_bytes: int, hardware: HardwareConfig, description: str
) -> None:
    if required_bytes > hardware.sram_bytes:
        raise ValueError(
            f"{description} requires {required_bytes} SRAM bytes, but hardware has "
            f"{hardware.sram_bytes}"
        )
