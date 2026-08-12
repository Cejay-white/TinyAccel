# TinyAccel

**A minimal AI compiler and accelerator simulator for learning how modern AI
hardware works.**

TinyAccel turns tensor and NHWC convolution graphs into tiled accelerator
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
Total cycles:       1722
DRAM bytes read:    26880
DRAM bytes written: 4608
Peak SRAM bytes:    6656
```

The simulator currently models instructions sequentially. Cycle counts are an
analytical estimate based on configurable DMA bandwidth and MAC throughput,
not a cycle-accurate hardware model.

`compiled.timeline()` renders the measured instruction events as a compact
ASCII Gantt chart. Large programs retain their first and last events while the
middle is folded:

```text
TinyAccel Instruction Timeline
cycles 0                                      1722
0000 ZERO      [     0,4     ) |#                                               |
0001 DMA_LOAD  [     4,20    ) |#                                               |
0002 DMA_LOAD  [    20,36    ) |##                                              |
     ... 96 instructions omitted ...
0099 DMA_LOAD  [  1682,1690  ) |                                              ##|
0100 MATMUL    [  1690,1706  ) |                                               #|
0101 DMA_STORE [  1706,1722  ) |                                               #|
```

## What the current prototype contains

- SSA-like graph IR with static shape and dtype validation
- Tensor layouts as canonical type metadata (`NHWC`, `NCHW`, `HWIO`, `OIHW`)
- Float32 NHWC-by-HWIO Conv2D with stride, padding, and dilation
- Constants, scalar/bias broadcasting, `add`, and `relu`
- Value producer/user queries and multi-output reference execution
- Canonical IR printing and validated text parsing
- NumPy reference executor for correctness checking
- Composable Pass Manager with per-pass IR snapshots
- Constant folding, algebraic simplification, and dead-code elimination
- MatMul + bias + ReLU operator fusion
- GraphViz DOT graph export
- Configurable M/N/K tiling
- Explicit spatial/reduction loop schedules used by ISA lowering
- Conv2D schedules over `N/H/W/OC` and reductions over `KH/KW/IC`
- SSA value lifetime analysis
- Alignment-aware linear-scan SRAM planning with buffer reuse
- Arena-backed simulation of planned intermediate tensors
- Multi-operator tiled lowering with broadcast-aware DMA slices
- Human-readable `ZERO`, `DMA_LOAD`, `MATMUL`, `ADD`, `RELU`, `CONV2D`, and
  `DMA_STORE` instructions
- Functional accelerator simulation checked against NumPy
- Cycle, DRAM traffic, and peak SRAM reporting
- Per-instruction cycle events and an ASCII execution timeline
- Hardware configuration and compile-time SRAM capacity checking

## Optimization pipeline

Compilation runs a deterministic optimization pipeline by default:

```text
Constant Folding
      -> Algebraic Simplification
      -> MatMul + Bias + ReLU Fusion
      -> Dead Code Elimination
```

Use `CompileOptions(optimize=False)` to inspect the unoptimized program. The
fusion example verifies both versions against NumPy and compares instructions,
cycles, and DRAM traffic:

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

Schedules describe spatial and reduction loops, their extents, tile sizes, and
tile counts. The memory planner computes each SSA result's live interval and
uses first-fit allocation to reuse aligned SRAM arena ranges whose lifetimes do
not overlap. Inputs and constants remain external/immutable; generated tensor
results are views into the planned arena used by the simulator.

## Run tests

```bash
python -m unittest discover -v
```

## Roadmap

- **v0.2:** multi-operator IR, foundational passes, fusion, and visualization
- **v0.3:** explicit schedules, lifetime analysis, and memory
  planning
- **v0.4 (current):** NHWC Conv2D from Graph IR through tiled ISA and simulation
- **v0.4 next:** NCHW, layout transformations, and im2col comparison
- **v0.5:** richer ISA, timelines, and resource-conflict simulation
- **v0.6:** PyTorch FX frontend and additional backends

TinyAccel is an educational and experimental project. It aims to make the
compiler-to-hardware path concrete, not to replace production compiler stacks.
