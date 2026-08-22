# TinyAccel

**A minimal AI compiler and accelerator simulator for learning how modern AI
hardware works.**

TinyAccel turns tensor and NCHW/NHWC convolution graphs into tiled accelerator
instructions, executes them on a small functional simulator, and explains the
cost in cycles and memory traffic. It stays intentionally small so the complete
path from a multi-operator graph to hardware-style execution remains
understandable.

```text
Tensor Graph -> Graph IR -> Schedule -> Memory Plan -> TinyAccel ISA -> Simulator
                                                                       |
                                               Result + cycle/memory report
```

## Quick start

TinyAccel requires Python 3.10+ and NumPy.

```bash
python -m pip install -e .
python -m examples.matmul
python -m examples.fusion
python -m examples.schedule_memory
python -m examples.conv2d
python -m examples.layout_transform
python -m examples.im2col
python -m examples.resource_overlap
python -m examples.double_buffer
```

The core API is small:

```python
import numpy as np
import tinyaccel

builder = tinyaccel.GraphBuilder()
a = builder.input("a", (48, 40))
b = builder.input("b", (40, 24))
c = builder.matmul(a, b)
graph = builder.build(c)

compiled = tinyaccel.compile(
    graph,
    options=tinyaccel.CompileOptions(tile_m=16, tile_n=16, tile_k=8),
)

a_data = np.ones((48, 40), dtype=np.float32)
b_data = np.ones((40, 24), dtype=np.float32)
result = compiled.run(a_data, b_data)
print(compiled.report())
print(compiled.timeline())
print(compiled.schedule)
print(compiled.memory_plan)
```

The canonical IR is also parseable, which makes it useful for debugging and
small compiler experiments:

```python
text = str(graph)
restored_graph = tinyaccel.parse_graph(text)
assert str(restored_graph) == text
```

Example report:

```text
TinyAccel Simulation Report
============================
ZERO       instructions=6    cycles=18
DMA_LOAD   instructions=60   cycles=840
MATMUL     instructions=30   cycles=720
DMA_STORE  instructions=6    cycles=144
----------------------------
Execution mode:     sequential
Total cycles:       1722
Sequential cycles:  1722
Overlap saved:      0
Layout cycles:      0
DRAM bytes read:    26880
DRAM bytes written: 4608
SRAM bytes read:    4608
SRAM bytes written: 26880
Peak SRAM bytes:    2048
DMA     busy:      984      utilization=57.1%
COMPUTE busy:      738      utilization=42.9%
LAYOUT  busy:      0        utilization=0.0%
```

The simulator defaults to sequential timing for compatibility and can also
schedule independent DMA, compute, and layout work concurrently. Cycle counts
are an analytical estimate based on configurable DMA bandwidth, MAC throughput,
and vector layout throughput, not a cycle-accurate hardware model.

`compiled.timeline()` renders the measured instruction events as a compact
ASCII Gantt chart. Large programs retain their first and last events while the
middle is folded:

```text
TinyAccel Instruction Timeline
cycles 0                                      1722
0000 COMPUTE ZERO      [     0,4     ) |#                                               |
0001 DMA     DMA_LOAD  [     4,20    ) |#                                               |
0002 DMA     DMA_LOAD  [    20,36    ) |##                                              |
     ... 96 instructions omitted ...
0099 DMA     DMA_LOAD  [  1682,1690  ) |                                              ##|
0100 COMPUTE MATMUL    [  1690,1706  ) |                                               #|
0101 DMA     DMA_STORE [  1706,1722  ) |                                               #|
```

## What the current prototype contains

- SSA-like graph IR with static shape and dtype validation
- Tensor layouts as canonical type metadata (`NHWC`, `NCHW`, `HWIO`, `OIHW`)
- Explicit tiled layout transforms (`NCHW` <-> `NHWC`, `OIHW` <-> `HWIO`)
- Float32 NHWC/HWIO and NCHW/OIHW Conv2D with stride, padding, and dilation
- Constants, scalar/bias broadcasting, `add`, and `relu`
- Value producer/user queries and multi-output reference execution
- Canonical IR printing and validated text parsing
- NumPy reference executor for correctness checking
- Composable Pass Manager with per-pass IR snapshots
- Constant folding, algebraic simplification, and dead-code elimination
- Conv2D layout canonicalization and redundant transform elimination
- MatMul + bias + ReLU operator fusion
- GraphViz DOT graph export
- Configurable MatMul M/N/K and Conv2D H/W/OC/IC tiling
- Explicit spatial/reduction loop schedules used by ISA lowering
- Conv2D schedules over `N/H/W/OC` and reductions over `KH/KW/IC`
- Partial Conv2D accumulation across configurable input-channel tiles
- Selectable direct or explicit im2col + MatMul Conv2D lowering
- SSA value lifetime analysis
- Alignment-aware linear-scan SRAM planning with buffer reuse
- Arena-backed simulation of planned intermediate tensors
- Multi-operator tiled lowering with broadcast-aware DMA slices
- Human-readable `ZERO`, `DMA_LOAD`, `IM2COL`, `RESHAPE`, `MATMUL`, `ADD`,
  `RELU`, `TRANSPOSE`, `CONV2D`, and `DMA_STORE` instructions
- Functional accelerator simulation checked against NumPy
- Cycle, DRAM/SRAM transfer traffic, and peak SRAM reporting
- Resource-tagged instruction events and an overlapping ASCII execution timeline
- Optional DMA/compute/layout resource-conflict and dependency simulation
- Optional ping/pong buffering and reduction-tile DMA prefetch
- Hardware configuration and compile-time SRAM capacity checking

## Optimization pipeline

Compilation runs a deterministic optimization pipeline by default:

```text
Conv2D Layout Canonicalization
      -> Constant Folding
      -> Algebraic Simplification
      -> Layout Transform Simplification
      -> MatMul + Bias + ReLU Fusion
      -> Dead Code Elimination
```

Use `CompileOptions(optimize=False)` to inspect the unoptimized program. NCHW
Conv2D compilation requires the default pipeline because the backend schedule
intentionally accepts only canonical NHWC/HWIO operations. The reference
executor supports both layout pairs directly. The fusion example verifies both
optimization modes against NumPy and compares instructions, cycles, and
DRAM/SRAM transfer traffic:

```bash
python -m examples.fusion
```

Passes can also be run independently:

```python
optimized, trace = tinyaccel.default_pipeline().run_with_trace(graph)
for result in trace:
    print(result.pass_name)
    print(result.graph)
```

Use `graph.to_dot()` to export either graph for GraphViz rendering.

## Scheduling and memory planning

Compilation retains the explicit forms between graph optimization and ISA
lowering. They can be inspected directly:

```python
print(compiled.schedule)
print(compiled.memory_plan)
```

Custom schedules can be passed directly to compilation. Graph optimization
must be disabled so the scheduled operations remain identical to the graph
being lowered:

```python
custom_schedule = tinyaccel.create_schedule(
    graph, tile_m=8, tile_n=8, tile_k=4
)
custom = tinyaccel.compile(
    graph,
    options=tinyaccel.CompileOptions(optimize=False),
    schedule=custom_schedule,
)
```

Schedules describe spatial and reduction loops, their extents, tile sizes, and
tile counts. The memory planner computes each SSA result's live interval and
uses first-fit allocation to reuse aligned SRAM arena ranges whose lifetimes do
not overlap. Inputs, constants, and final graph outputs reside in DRAM;
non-output intermediate results are views into the SRAM arena used by the
simulator. Pass `materialize_outputs=True` to `plan_memory` for standalone
experiments that also retain graph outputs in the arena.

DMA instructions expose the persistent value's memory space as `space=DRAM` or
`space=SRAM`. Reported SRAM traffic covers the DMA-side movement through local
tile buffers and the planned arena; it does not yet include compute-unit SRAM
accesses made internally by MatMul, Conv2D, Add, ReLU, Transpose, or Im2col.

For Conv2D, `CompileOptions(tile_ic=...)` controls reduction tiling over input
channels. Each output tile is zeroed once, accumulates all IC partial results,
and is stored once. This allows high-channel convolutions to run within a small
SRAM capacity without changing graph semantics.

Schedule construction validates the exact axis order, extent, and
spatial/reduction kind required by each operation. Tiles must not exceed their
loop extents, and scheduled operations must match graph program order.

## Layout transformations

`layout_transform` makes physical layout changes explicit in the graph. It
infers the permuted shape, preserves dtype, and carries the target layout in
both the operation attributes and output type:

```python
builder = tinyaccel.GraphBuilder()
input_nchw = builder.input("input", (1, 3, 8, 8), layout="NCHW")
input_nhwc = builder.layout_transform(input_nchw, "NHWC")
graph = builder.build(input_nhwc)
```

Conv2D can also be authored directly in NCHW/OIHW form. The default pipeline
inserts NCHW-to-NHWC and OIHW-to-HWIO transforms, schedules the canonical
convolution, then transforms the result back to NCHW:

```python
builder = tinyaccel.GraphBuilder()
input_nchw = builder.input("input", (1, 3, 8, 8), layout="NCHW")
weight_oihw = builder.input("weight", (16, 3, 3, 3), layout="OIHW")
output_nchw = builder.conv2d(input_nchw, weight_oihw, padding=1)
compiled = tinyaccel.compile(builder.build(output_nchw))
```

The reference executor uses the same checked axis permutation as the compiler.
The backend tiles target-layout axes, loads the corresponding source tile,
emits `TRANSPOSE`, and stores the target tile. Constant transforms are folded
and adjacent inverse transforms are eliminated by the default optimization
pipeline. See the automatic end-to-end NCHW/OIHW Conv2D path with:

```bash
python -m examples.layout_transform
```

## Direct vs im2col Conv2D

Conv2D defaults to the direct tiled `CONV2D` instruction. Set
`conv2d_lowering="im2col"` to materialize each input patch tile and execute the
same convolution through `MATMUL`:

```python
im2col = tinyaccel.compile(
    graph,
    options=tinyaccel.CompileOptions(conv2d_lowering="im2col"),
)
```

Both paths use the same spatial/reduction schedule, padding semantics, IC
tiling, DMA slices, and numerical reference. The im2col path emits
`IM2COL -> RESHAPE -> MATMUL`, accounts for the materialized columns in its
compile-time SRAM requirement, and treats `RESHAPE` as a zero-cycle view.

`HardwareConfig(vector_elements_per_cycle=...)` controls `TRANSPOSE` and
`IM2COL` throughput. `SimulationReport.layout_cycles` aggregates the cycles
spent in `TRANSPOSE`, `IM2COL`, and `RESHAPE`, making layout overhead explicit
beside total cycles and peak SRAM. Run the side-by-side comparison with:

```bash
python -m examples.im2col
```

## Resource-aware timing

Sequential timing remains the default, so existing cycle comparisons do not
change. Enable independent DMA, compute, and layout lanes in the analytical
scheduler with:

```python
hardware = tinyaccel.HardwareConfig(overlap_resources=True)
compiled = tinyaccel.compile(graph, hardware=hardware)
```

The scheduler preserves RAW, WAR, and WAW dependencies for local buffers and
persistent values while serializing instructions that use the same hardware
resource. Work without a dependency may overlap on different resources. The
functional simulator still executes deterministically in program order; only
the analytical start/end cycles are scheduled concurrently.

`SimulationReport.sequential_cycles` reports the total instruction work,
`total_cycles` reports elapsed time, and `overlap_cycles_saved` exposes the
difference. `resource_cycles` and `resource_utilization` break down activity for
the `DMA`, `COMPUTE`, and `LAYOUT` lanes. Timeline rows include their assigned
resource, so overlapping intervals and resource stalls remain visible:

```bash
python -m examples.resource_overlap
```

## Double buffering and tile prefetch

Resource overlap alone cannot hide reduction-tile DMA when every tile reuses
the same local operand names: the scheduler must preserve the live tile until
its compute finishes. Enable explicit ping/pong buffers with:

```python
compiled = tinyaccel.compile(
    graph,
    options=tinyaccel.CompileOptions(double_buffer=True),
    hardware=tinyaccel.HardwareConfig(overlap_resources=True),
)
```

MatMul alternates `lhs_0/rhs_0` and `lhs_1/rhs_1`; direct and im2col Conv2D
similarly alternate their input, weight, and im2col-column buffers across K or
IC reduction tiles. This lets the DMA unit prefetch the next tile while the
compute unit consumes the current one. Fused MatMul + bias + ReLU uses the same
pipeline and safely reuses one RHS slot for the final bias load.

Double buffering keeps instruction work and external traffic unchanged but
uses more SRAM. The extra operand and im2col-column slots participate in the
normal compile-time capacity check. A reduction with only one tile
automatically falls back to one buffer. Run the SRAM/latency comparison with:

```bash
python -m examples.double_buffer
```

## Run tests

```bash
python -m unittest discover -v
```

## Roadmap

- **v0.2:** multi-operator IR, foundational passes, fusion, and visualization
- **v0.3:** explicit schedules, lifetime analysis, and memory
  planning
- **v0.4:** NCHW/NHWC Conv2D, layout cost modeling, automatic
  canonicalization, and direct/im2col lowering comparison
- **v0.5 (current):** resource-tagged timelines, dependency hazards,
  DMA/compute/layout overlap, and reduction-tile ping/pong prefetch
- **v0.6:** PyTorch FX frontend and additional backends

TinyAccel is an educational and experimental project. It aims to make the
compiler-to-hardware path concrete, not to replace production compiler stacks.
