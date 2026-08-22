"""Lower TinyAccel graph IR into the minimal accelerator ISA."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING, Iterable

import numpy as np

from .hardware import HardwareConfig
from .ir import Graph, TensorType, layout_permutation
from .isa import Instruction, MemorySpace, Opcode, Program
from .memory import MemoryPlan, plan_memory
from .passes import default_pipeline
from .schedule import Schedule, ScheduledOperation, create_schedule

if TYPE_CHECKING:
    from .simulator import SimulationReport


@dataclass(frozen=True)
class CompileOptions:
    """Optimization and tiling choices made by the compiler."""

    tile_m: int = 32
    tile_n: int = 32
    tile_k: int = 32
    tile_h: int = 8
    tile_w: int = 8
    tile_oc: int = 16
    optimize: bool = True
    tile_ic: int = 16
    conv2d_lowering: str = "direct"
    double_buffer: bool = False

    def __post_init__(self) -> None:
        for name, value in (
            ("tile_m", self.tile_m),
            ("tile_n", self.tile_n),
            ("tile_k", self.tile_k),
            ("tile_h", self.tile_h),
            ("tile_w", self.tile_w),
            ("tile_oc", self.tile_oc),
            ("tile_ic", self.tile_ic),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.conv2d_lowering not in {"direct", "im2col"}:
            raise ValueError(
                "conv2d_lowering must be 'direct' or 'im2col', got "
                f"{self.conv2d_lowering!r}"
            )
        if not isinstance(self.double_buffer, bool):
            raise TypeError(
                "double_buffer must be a bool, got "
                f"{type(self.double_buffer).__name__}"
            )


class Executable:
    """A compiled program that can be run by the TinyAccel simulator."""

    def __init__(
        self,
        program: Program,
        hardware: HardwareConfig,
        graph: Graph,
        schedule: Schedule,
        memory_plan: MemoryPlan,
    ) -> None:
        self.program = program
        self.hardware = hardware
        self.graph = graph
        self.schedule = schedule
        self.memory_plan = memory_plan
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
    schedule: Schedule | None = None,
) -> Executable:
    """Compile a supported graph into a tiled TinyAccel program."""

    options = options or CompileOptions()
    hardware = hardware or HardwareConfig()
    graph.validate()
    if schedule is not None and options.optimize:
        raise ValueError("custom schedule requires CompileOptions(optimize=False)")
    lowered_graph = default_pipeline().run(graph) if options.optimize else graph

    if len(lowered_graph.outputs) != 1:
        raise NotImplementedError("accelerator backend currently requires one output")
    for value in lowered_graph.values:
        if value.type.dtype != np.dtype("float32"):
            raise NotImplementedError("accelerator backend currently supports float32")

    if schedule is None:
        schedule = create_schedule(
            lowered_graph,
            tile_m=options.tile_m,
            tile_n=options.tile_n,
            tile_k=options.tile_k,
            tile_h=options.tile_h,
            tile_w=options.tile_w,
            tile_oc=options.tile_oc,
            tile_ic=options.tile_ic,
        )
    elif schedule.graph is not lowered_graph:
        raise ValueError("custom schedule must target the graph being compiled")
    memory_plan = plan_memory(
        lowered_graph, capacity_bytes=hardware.sram_bytes
    )
    instructions: list[Instruction] = []
    constants: dict[str, np.ndarray] = {}
    for scheduled in schedule.operations:
        operation = scheduled.operation
        if operation.op == "constant":
            constants[operation.output.name] = np.asarray(
                operation.attributes["value"], dtype=operation.output.type.dtype
            ).copy()
        elif operation.op == "matmul":
            _lower_matmul(
                scheduled,
                instructions,
                hardware,
                memory_plan.total_bytes,
                options.double_buffer,
            )
        elif operation.op == "add":
            _lower_add(scheduled, instructions, hardware, memory_plan.total_bytes)
        elif operation.op == "relu":
            _lower_relu(scheduled, instructions, hardware, memory_plan.total_bytes)
        elif operation.op == "layout_transform":
            _lower_layout_transform(
                scheduled, instructions, hardware, memory_plan.total_bytes
            )
        elif operation.op == "matmul_bias_relu":
            _lower_matmul_bias_relu(
                scheduled,
                instructions,
                hardware,
                memory_plan.total_bytes,
                options.double_buffer,
            )
        elif operation.op == "conv2d":
            if options.conv2d_lowering == "direct":
                _lower_conv2d(
                    scheduled,
                    instructions,
                    hardware,
                    memory_plan.total_bytes,
                    options.double_buffer,
                )
            else:
                _lower_conv2d_im2col(
                    scheduled,
                    instructions,
                    hardware,
                    memory_plan.total_bytes,
                    options.double_buffer,
                )
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
    value_spaces = {
        value.name: (
            MemorySpace.SRAM
            if value.name in memory_plan.allocations
            else MemorySpace.DRAM
        )
        for value in lowered_graph.values
    }
    instructions = _annotate_dma_spaces(instructions, value_spaces)
    output = lowered_graph.outputs[0]
    program = Program(
        instructions=tuple(instructions),
        input_types=input_types,
        output_name=output.name,
        output_shape=output.type.shape,
        output_dtype=output.type.dtype,
        value_types=value_types,
        value_spaces=value_spaces,
        constants=constants,
        memory_plan=memory_plan,
    )
    return Executable(program, hardware, lowered_graph, schedule, memory_plan)


def _lower_matmul(
    scheduled: ScheduledOperation,
    instructions: list[Instruction],
    hardware: HardwareConfig,
    reserved_sram_bytes: int,
    double_buffer: bool,
) -> None:
    operation = scheduled.operation
    lhs, rhs = operation.inputs
    output = operation.output
    m_size, k_size = lhs.type.shape
    _, n_size = rhs.type.shape

    tile_m = scheduled.loop("m").tile
    tile_n = scheduled.loop("n").tile
    tile_k = scheduled.loop("k").tile
    reduction_extents = _tile_extents(k_size, tile_k)
    buffer_slots = _reduction_buffer_slots(
        scheduled.loop("k").tiles, double_buffer
    )
    buffered_k_extent = _max_buffered_reduction_extent(
        reduction_extents, buffer_slots
    )
    for m_offset in range(0, m_size, tile_m):
        m_extent = min(tile_m, m_size - m_offset)
        for n_offset in range(0, n_size, tile_n):
            n_extent = min(tile_n, n_size - n_offset)
            _require_sram(
                (
                    buffered_k_extent * (m_extent + n_extent)
                    + m_extent * n_extent
                )
                * output.type.dtype.itemsize,
                hardware,
                "matmul tile",
                reserved_sram_bytes,
            )
            instructions.append(
                Instruction(
                    Opcode.ZERO,
                    {"buffer": "acc", "shape": (m_extent, n_extent)},
                )
            )
            for reduction_index, k_offset in enumerate(
                range(0, k_size, tile_k)
            ):
                k_extent = min(tile_k, k_size - k_offset)
                lhs_buffer = _reduction_buffer("lhs", reduction_index, buffer_slots)
                rhs_buffer = _reduction_buffer("rhs", reduction_index, buffer_slots)
                instructions.extend(
                    (
                        _load(
                            lhs.name,
                            lhs_buffer,
                            (m_offset, k_offset),
                            (m_extent, k_extent),
                        ),
                        _load(
                            rhs.name,
                            rhs_buffer,
                            (k_offset, n_offset),
                            (k_extent, n_extent),
                        ),
                        Instruction(
                            Opcode.MATMUL,
                            {
                                "lhs": lhs_buffer,
                                "rhs": rhs_buffer,
                                "accumulator": "acc",
                            },
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
    scheduled: ScheduledOperation,
    instructions: list[Instruction],
    hardware: HardwareConfig,
    reserved_sram_bytes: int,
) -> None:
    operation = scheduled.operation
    lhs, rhs = operation.inputs
    output = operation.output
    for offset, shape in _output_tiles(output.type, scheduled):
        lhs_offset, lhs_shape = _broadcast_slice(
            lhs.type.shape, output.type.shape, offset, shape
        )
        rhs_offset, rhs_shape = _broadcast_slice(
            rhs.type.shape, output.type.shape, offset, shape
        )
        required = (
            _element_count(lhs_shape)
            + _element_count(rhs_shape)
            + _element_count(shape)
        ) * output.type.dtype.itemsize
        _require_sram(required, hardware, "add tile", reserved_sram_bytes)
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
    scheduled: ScheduledOperation,
    instructions: list[Instruction],
    hardware: HardwareConfig,
    reserved_sram_bytes: int,
) -> None:
    operation = scheduled.operation
    source = operation.inputs[0]
    output = operation.output
    for offset, shape in _output_tiles(output.type, scheduled):
        required = 2 * _element_count(shape) * output.type.dtype.itemsize
        _require_sram(required, hardware, "relu tile", reserved_sram_bytes)
        instructions.extend(
            (
                _load(source.name, "lhs", offset, shape),
                Instruction(Opcode.RELU, {"input": "lhs", "output": "acc"}),
                _store("acc", output.name, offset, shape),
            )
        )


def _lower_layout_transform(
    scheduled: ScheduledOperation,
    instructions: list[Instruction],
    hardware: HardwareConfig,
    reserved_sram_bytes: int,
) -> None:
    operation = scheduled.operation
    source = operation.inputs[0]
    output = operation.output
    permutation = layout_permutation(source.type.layout, output.type.layout)
    for output_offset, output_shape in _output_tiles(output.type, scheduled):
        input_offset = [0] * len(permutation)
        input_shape = [0] * len(permutation)
        for output_axis, input_axis in enumerate(permutation):
            input_offset[input_axis] = output_offset[output_axis]
            input_shape[input_axis] = output_shape[output_axis]
        required = 2 * _element_count(output_shape) * output.type.dtype.itemsize
        _require_sram(
            required,
            hardware,
            "layout_transform tile",
            reserved_sram_bytes,
        )
        instructions.extend(
            (
                _load(
                    source.name,
                    "input",
                    tuple(input_offset),
                    tuple(input_shape),
                ),
                Instruction(
                    Opcode.TRANSPOSE,
                    {
                        "input": "input",
                        "output": "acc",
                        "permutation": permutation,
                    },
                ),
                _store("acc", output.name, output_offset, output_shape),
            )
        )


def _lower_matmul_bias_relu(
    scheduled: ScheduledOperation,
    instructions: list[Instruction],
    hardware: HardwareConfig,
    reserved_sram_bytes: int,
    double_buffer: bool,
) -> None:
    operation = scheduled.operation
    lhs, rhs, bias = operation.inputs
    output = operation.output
    m_size, k_size = lhs.type.shape
    _, n_size = rhs.type.shape

    tile_m = scheduled.loop("m").tile
    tile_n = scheduled.loop("n").tile
    tile_k = scheduled.loop("k").tile
    reduction_extents = _tile_extents(k_size, tile_k)
    buffer_slots = _reduction_buffer_slots(
        scheduled.loop("k").tiles, double_buffer
    )
    buffered_k_extent = _max_buffered_reduction_extent(
        reduction_extents, buffer_slots
    )
    final_slot_extents = _final_reduction_slot_extents(
        reduction_extents, buffer_slots
    )
    for m_offset in range(0, m_size, tile_m):
        m_extent = min(tile_m, m_size - m_offset)
        for n_offset in range(0, n_size, tile_n):
            n_extent = min(tile_n, n_size - n_offset)
            bias_offset, bias_shape = _broadcast_slice(
                bias.type.shape,
                output.type.shape,
                (m_offset, n_offset),
                (m_extent, n_extent),
            )
            required = max(
                (
                    buffered_k_extent * (m_extent + n_extent)
                    + m_extent * n_extent
                )
                * output.type.dtype.itemsize,
                (
                    sum(final_slot_extents) * m_extent
                    + sum(final_slot_extents[1:]) * n_extent
                    + _element_count(bias_shape)
                    + m_extent * n_extent
                )
                * output.type.dtype.itemsize,
            )
            _require_sram(
                required, hardware, "fused matmul tile", reserved_sram_bytes
            )
            instructions.append(
                Instruction(
                    Opcode.ZERO,
                    {"buffer": "acc", "shape": (m_extent, n_extent)},
                )
            )
            for reduction_index, k_offset in enumerate(
                range(0, k_size, tile_k)
            ):
                k_extent = min(tile_k, k_size - k_offset)
                lhs_buffer = _reduction_buffer("lhs", reduction_index, buffer_slots)
                rhs_buffer = _reduction_buffer("rhs", reduction_index, buffer_slots)
                instructions.extend(
                    (
                        _load(
                            lhs.name,
                            lhs_buffer,
                            (m_offset, k_offset),
                            (m_extent, k_extent),
                        ),
                        _load(
                            rhs.name,
                            rhs_buffer,
                            (k_offset, n_offset),
                            (k_extent, n_extent),
                        ),
                        Instruction(
                            Opcode.MATMUL,
                            {
                                "lhs": lhs_buffer,
                                "rhs": rhs_buffer,
                                "accumulator": "acc",
                            },
                        ),
                    )
                )
            bias_buffer = _reduction_buffer("rhs", 0, buffer_slots)
            instructions.extend(
                (
                    _load(bias.name, bias_buffer, bias_offset, bias_shape),
                    Instruction(
                        Opcode.ADD,
                        {
                            "lhs": "acc",
                            "rhs": bias_buffer,
                            "output": "acc",
                        },
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


def _lower_conv2d(
    scheduled: ScheduledOperation,
    instructions: list[Instruction],
    hardware: HardwareConfig,
    reserved_sram_bytes: int,
    double_buffer: bool,
) -> None:
    operation = scheduled.operation
    input_value, weight = operation.inputs
    output = operation.output
    _, _, _, input_c = input_value.type.shape
    kernel_h, kernel_w, _, _ = weight.type.shape
    stride_h, stride_w = operation.attributes["stride"]
    pad_top, _, pad_left, _ = operation.attributes["padding"]
    dilation_h, dilation_w = operation.attributes["dilation"]
    effective_h = (kernel_h - 1) * dilation_h + 1
    effective_w = (kernel_w - 1) * dilation_w + 1
    tile_h = scheduled.loop("h").tile
    tile_w = scheduled.loop("w").tile
    tile_oc = scheduled.loop("oc").tile
    tile_ic = scheduled.loop("ic").tile
    reduction_extents = _tile_extents(input_c, tile_ic)
    buffer_slots = _reduction_buffer_slots(
        scheduled.loop("ic").tiles, double_buffer
    )
    buffered_ic_extent = _max_buffered_reduction_extent(
        reduction_extents, buffer_slots
    )
    n_size, output_h, output_w, output_c = output.type.shape

    for n_offset in range(n_size):
        for h_offset in range(0, output_h, tile_h):
            h_extent = min(tile_h, output_h - h_offset)
            input_h_extent = (h_extent - 1) * stride_h + effective_h
            input_h_offset = h_offset * stride_h - pad_top
            for w_offset in range(0, output_w, tile_w):
                w_extent = min(tile_w, output_w - w_offset)
                input_w_extent = (w_extent - 1) * stride_w + effective_w
                input_w_offset = w_offset * stride_w - pad_left
                for oc_offset in range(0, output_c, tile_oc):
                    oc_extent = min(tile_oc, output_c - oc_offset)
                    output_shape = (1, h_extent, w_extent, oc_extent)
                    required = (
                        buffered_ic_extent
                        * (
                            input_h_extent * input_w_extent
                            + kernel_h * kernel_w * oc_extent
                        )
                        + _element_count(output_shape)
                    ) * output.type.dtype.itemsize
                    _require_sram(
                        required,
                        hardware,
                        "conv2d tile",
                        reserved_sram_bytes,
                    )
                    instructions.append(
                        Instruction(
                            Opcode.ZERO,
                            {"buffer": "acc", "shape": output_shape},
                        )
                    )
                    for reduction_index, ic_offset in enumerate(
                        range(0, input_c, tile_ic)
                    ):
                        ic_extent = min(tile_ic, input_c - ic_offset)
                        input_buffer = _reduction_buffer(
                            "input", reduction_index, buffer_slots
                        )
                        weight_buffer = _reduction_buffer(
                            "weight", reduction_index, buffer_slots
                        )
                        input_shape = (
                            1,
                            input_h_extent,
                            input_w_extent,
                            ic_extent,
                        )
                        weight_shape = (
                            kernel_h,
                            kernel_w,
                            ic_extent,
                            oc_extent,
                        )
                        instructions.extend(
                            (
                                _load(
                                    input_value.name,
                                    input_buffer,
                                    (
                                        n_offset,
                                        input_h_offset,
                                        input_w_offset,
                                        ic_offset,
                                    ),
                                    input_shape,
                                    padded=True,
                                ),
                                _load(
                                    weight.name,
                                    weight_buffer,
                                    (0, 0, ic_offset, oc_offset),
                                    weight_shape,
                                ),
                                Instruction(
                                    Opcode.CONV2D,
                                    {
                                        "input": input_buffer,
                                        "weight": weight_buffer,
                                        "accumulator": "acc",
                                        "stride": (stride_h, stride_w),
                                        "dilation": (dilation_h, dilation_w),
                                    },
                                ),
                            )
                        )
                    instructions.append(
                        _store(
                            "acc",
                            output.name,
                            (n_offset, h_offset, w_offset, oc_offset),
                            output_shape,
                        )
                    )


def _lower_conv2d_im2col(
    scheduled: ScheduledOperation,
    instructions: list[Instruction],
    hardware: HardwareConfig,
    reserved_sram_bytes: int,
    double_buffer: bool,
) -> None:
    operation = scheduled.operation
    input_value, weight = operation.inputs
    output = operation.output
    _, _, _, input_c = input_value.type.shape
    kernel_h, kernel_w, _, _ = weight.type.shape
    stride_h, stride_w = operation.attributes["stride"]
    pad_top, _, pad_left, _ = operation.attributes["padding"]
    dilation_h, dilation_w = operation.attributes["dilation"]
    effective_h = (kernel_h - 1) * dilation_h + 1
    effective_w = (kernel_w - 1) * dilation_w + 1
    tile_h = scheduled.loop("h").tile
    tile_w = scheduled.loop("w").tile
    tile_oc = scheduled.loop("oc").tile
    tile_ic = scheduled.loop("ic").tile
    reduction_extents = _tile_extents(input_c, tile_ic)
    buffer_slots = _reduction_buffer_slots(
        scheduled.loop("ic").tiles, double_buffer
    )
    buffered_ic_extent = _max_buffered_reduction_extent(
        reduction_extents, buffer_slots
    )
    n_size, output_h, output_w, output_c = output.type.shape

    for n_offset in range(n_size):
        for h_offset in range(0, output_h, tile_h):
            h_extent = min(tile_h, output_h - h_offset)
            input_h_extent = (h_extent - 1) * stride_h + effective_h
            input_h_offset = h_offset * stride_h - pad_top
            for w_offset in range(0, output_w, tile_w):
                w_extent = min(tile_w, output_w - w_offset)
                input_w_extent = (w_extent - 1) * stride_w + effective_w
                input_w_offset = w_offset * stride_w - pad_left
                for oc_offset in range(0, output_c, tile_oc):
                    oc_extent = min(tile_oc, output_c - oc_offset)
                    output_shape = (1, h_extent, w_extent, oc_extent)
                    accumulator_shape = (h_extent * w_extent, oc_extent)
                    required = (
                        buffered_ic_extent
                        * (
                            input_h_extent * input_w_extent
                            + kernel_h * kernel_w * oc_extent
                            + h_extent * w_extent * kernel_h * kernel_w
                        )
                        + _element_count(accumulator_shape)
                    ) * output.type.dtype.itemsize
                    _require_sram(
                        required,
                        hardware,
                        "im2col conv2d tile",
                        reserved_sram_bytes,
                    )
                    instructions.append(
                        Instruction(
                            Opcode.ZERO,
                            {"buffer": "acc", "shape": accumulator_shape},
                        )
                    )
                    for reduction_index, ic_offset in enumerate(
                        range(0, input_c, tile_ic)
                    ):
                        ic_extent = min(tile_ic, input_c - ic_offset)
                        input_buffer = _reduction_buffer(
                            "input", reduction_index, buffer_slots
                        )
                        columns_buffer = _reduction_buffer(
                            "lhs", reduction_index, buffer_slots
                        )
                        weight_buffer = _reduction_buffer(
                            "weight", reduction_index, buffer_slots
                        )
                        input_shape = (
                            1,
                            input_h_extent,
                            input_w_extent,
                            ic_extent,
                        )
                        weight_shape = (
                            kernel_h,
                            kernel_w,
                            ic_extent,
                            oc_extent,
                        )
                        columns_shape = (
                            h_extent * w_extent,
                            kernel_h * kernel_w * ic_extent,
                        )
                        instructions.extend(
                            (
                                _load(
                                    input_value.name,
                                    input_buffer,
                                    (
                                        n_offset,
                                        input_h_offset,
                                        input_w_offset,
                                        ic_offset,
                                    ),
                                    input_shape,
                                    padded=True,
                                ),
                                Instruction(
                                    Opcode.IM2COL,
                                    {
                                        "input": input_buffer,
                                        "output": columns_buffer,
                                        "kernel": (kernel_h, kernel_w),
                                        "stride": (stride_h, stride_w),
                                        "dilation": (dilation_h, dilation_w),
                                        "output_shape": (h_extent, w_extent),
                                    },
                                ),
                                _load(
                                    weight.name,
                                    weight_buffer,
                                    (0, 0, ic_offset, oc_offset),
                                    weight_shape,
                                ),
                                Instruction(
                                    Opcode.RESHAPE,
                                    {
                                        "input": weight_buffer,
                                        "output": weight_buffer,
                                        "shape": (
                                            columns_shape[1],
                                            oc_extent,
                                        ),
                                    },
                                ),
                                Instruction(
                                    Opcode.MATMUL,
                                    {
                                        "lhs": columns_buffer,
                                        "rhs": weight_buffer,
                                        "accumulator": "acc",
                                    },
                                ),
                            )
                        )
                    instructions.extend(
                        (
                            Instruction(
                                Opcode.RESHAPE,
                                {
                                    "input": "acc",
                                    "output": "acc",
                                    "shape": output_shape,
                                },
                            ),
                            _store(
                                "acc",
                                output.name,
                                (n_offset, h_offset, w_offset, oc_offset),
                                output_shape,
                            ),
                        )
                    )


def _output_tiles(
    tensor_type: TensorType, scheduled: ScheduledOperation
) -> Iterable[tuple[tuple[int, ...], tuple[int, ...]]]:
    shape = tensor_type.shape
    if not shape:
        yield (), ()
        return
    tile_shape = [loop.tile for loop in scheduled.loops]
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
    *,
    padded: bool = False,
) -> Instruction:
    operands = {"source": source, "buffer": buffer, "offset": offset, "shape": shape}
    if padded:
        operands["padded"] = True
    return Instruction(Opcode.DMA_LOAD, operands)


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


def _reduction_buffer_slots(reduction_tiles: int, double_buffer: bool) -> int:
    return 2 if double_buffer and reduction_tiles > 1 else 1


def _tile_extents(extent: int, tile: int) -> tuple[int, ...]:
    return tuple(
        min(tile, extent - offset) for offset in range(0, extent, tile)
    )


def _max_buffered_reduction_extent(
    reduction_extents: tuple[int, ...], slots: int
) -> int:
    if slots == 1:
        return max(reduction_extents)
    return max(
        current + previous
        for previous, current in zip(
            reduction_extents, reduction_extents[1:], strict=False
        )
    )


def _final_reduction_slot_extents(
    reduction_extents: tuple[int, ...], slots: int
) -> tuple[int, ...]:
    final = [0] * slots
    for index, extent in enumerate(reduction_extents):
        final[index % slots] = extent
    return tuple(final)


def _reduction_buffer(base: str, reduction_index: int, slots: int) -> str:
    if slots == 1:
        return base
    return f"{base}_{reduction_index % slots}"


def _annotate_dma_spaces(
    instructions: list[Instruction],
    value_spaces: dict[str, MemorySpace],
) -> list[Instruction]:
    annotated: list[Instruction] = []
    for instruction in instructions:
        if instruction.opcode is Opcode.DMA_LOAD:
            value_name = instruction.operands["source"]
        elif instruction.opcode is Opcode.DMA_STORE:
            value_name = instruction.operands["output"]
        else:
            annotated.append(instruction)
            continue
        operands = dict(instruction.operands)
        operands["space"] = value_spaces[value_name].value
        annotated.append(Instruction(instruction.opcode, operands))
    return annotated


def _require_sram(
    required_bytes: int,
    hardware: HardwareConfig,
    description: str,
    reserved_bytes: int = 0,
) -> None:
    total_bytes = reserved_bytes + required_bytes
    if total_bytes > hardware.sram_bytes:
        raise ValueError(
            f"{description} requires {required_bytes} SRAM bytes plus "
            f"{reserved_bytes} planned bytes ({total_bytes} total), but hardware "
            f"has {hardware.sram_bytes}"
        )
