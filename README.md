# TinyAccel

**A minimal AI compiler and accelerator simulator for learning how modern AI
hardware works.**

TinyAccel turns a tensor graph into tiled accelerator instructions, executes
them on a small functional simulator, and explains the cost in cycles and
memory traffic. The project intentionally starts with one operation—matrix
multiplication—so the complete stack stays visible and understandable.

```text
Tensor Graph -> Graph IR -> Tiled Lowering -> TinyAccel ISA -> Simulator
                                                              |
                                      Result + cycle/memory report
```

## Quick start

TinyAccel requires Python 3.10+ and NumPy.

```bash
python -m pip install -e .
python -m examples.matmul
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
Peak SRAM bytes:    2048
```

The simulator currently models instructions sequentially. Cycle counts are an
analytical estimate based on configurable DMA bandwidth and MAC throughput—not
a cycle-accurate hardware model.

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
- Canonical IR printing and validated text parsing
- Rank-2 `matmul` shape inference
- Configurable M/N/K tiling
- Human-readable `ZERO`, `DMA_LOAD`, `MATMUL`, and `DMA_STORE` instructions
- Functional accelerator simulation checked against NumPy
- Cycle, DRAM traffic, and peak SRAM reporting
- Per-instruction cycle events and an ASCII execution timeline
- Hardware configuration and compile-time SRAM capacity checking

## Run tests

```bash
python -m unittest discover -v
```

## Roadmap

- **v0.2:** more operators, constants, and graph visualization
- **v0.3:** optimization passes and operator fusion
- **v0.4:** scheduling and memory planning
- **v0.5:** richer ISA, timelines, and resource-conflict simulation
- **v0.6:** PyTorch FX frontend and additional backends

TinyAccel is an educational and experimental project. It aims to make the
compiler-to-hardware path concrete, not to replace production compiler stacks.
