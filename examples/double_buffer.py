"""Compare single-buffered and ping/pong-prefetched MatMul timing."""

import numpy as np

import tinyaccel
from tinyaccel.isa import Opcode


builder = tinyaccel.GraphBuilder()
lhs = builder.input("lhs", (4, 12))
rhs = builder.input("rhs", (12, 4))
graph = builder.build(builder.matmul(lhs, rhs, name="result"))
hardware = tinyaccel.HardwareConfig(
    dma_bytes_per_cycle=4,
    macs_per_cycle=1,
    overlap_resources=True,
)
common_options = dict(tile_m=4, tile_n=4, tile_k=4)
single = tinyaccel.compile(
    graph,
    options=tinyaccel.CompileOptions(**common_options),
    hardware=hardware,
)
ping_pong = tinyaccel.compile(
    graph,
    options=tinyaccel.CompileOptions(**common_options, double_buffer=True),
    hardware=hardware,
)

rng = np.random.default_rng(67)
lhs_data = rng.standard_normal(lhs.type.shape, dtype=np.float32)
rhs_data = rng.standard_normal(rhs.type.shape, dtype=np.float32)
expected = lhs_data @ rhs_data
single_result = single.run(lhs_data, rhs_data)
ping_pong_result = ping_pong.run(lhs_data, rhs_data)
np.testing.assert_allclose(single_result, expected, rtol=1e-5, atol=1e-5)
np.testing.assert_allclose(ping_pong_result, expected, rtol=1e-5, atol=1e-5)

print("Double-buffered reduction pipeline")
print("=" * 58)
print(f"{'Metric':<24} {'Single buffer':>14} {'Ping/pong':>14}")
print("-" * 58)
rows = (
    (
        "Sequential work",
        single.last_report.sequential_cycles,
        ping_pong.last_report.sequential_cycles,
    ),
    (
        "Elapsed cycles",
        single.last_report.total_cycles,
        ping_pong.last_report.total_cycles,
    ),
    (
        "Overlap saved",
        single.last_report.overlap_cycles_saved,
        ping_pong.last_report.overlap_cycles_saved,
    ),
    (
        "Peak SRAM bytes",
        single.last_report.peak_sram_bytes,
        ping_pong.last_report.peak_sram_bytes,
    ),
    (
        "DRAM bytes read",
        single.last_report.dram_bytes_read,
        ping_pong.last_report.dram_bytes_read,
    ),
)
for label, single_value, ping_pong_value in rows:
    print(f"{label:<24} {single_value:>14} {ping_pong_value:>14}")

load_buffers = [
    instruction.operands["buffer"]
    for instruction in ping_pong.program.instructions
    if instruction.opcode is Opcode.DMA_LOAD
]
print()
print("DMA buffer sequence:", " -> ".join(load_buffers))
print()
print(ping_pong.timeline(width=44, max_events=16))
