"""Compare unfused and fused MatMul + bias + ReLU execution."""

import numpy as np

import tinyaccel


builder = tinyaccel.GraphBuilder()
lhs = builder.input("lhs", (32, 24))
rhs = builder.input("rhs", (24, 16))
bias = builder.constant(np.linspace(-1, 1, 16, dtype=np.float32), name="bias")
product = builder.matmul(lhs, rhs, name="product")
biased = builder.add(product, bias, name="biased")
graph = builder.build(builder.relu(biased, name="result"))

options = {"tile_m": 8, "tile_n": 8, "tile_k": 8}
unfused = tinyaccel.compile(
    graph, options=tinyaccel.CompileOptions(**options, optimize=False)
)
fused = tinyaccel.compile(
    graph, options=tinyaccel.CompileOptions(**options, optimize=True)
)

rng = np.random.default_rng(1)
lhs_data = rng.standard_normal((32, 24), dtype=np.float32)
rhs_data = rng.standard_normal((24, 16), dtype=np.float32)
expected = tinyaccel.evaluate(graph, lhs_data, rhs_data)
unfused_result = unfused.run(lhs_data, rhs_data)
fused_result = fused.run(lhs_data, rhs_data)

np.testing.assert_allclose(unfused_result, expected, rtol=1e-5, atol=1e-5)
np.testing.assert_allclose(fused_result, expected, rtol=1e-5, atol=1e-5)

print("Original IR")
print("-----------")
print(graph)
print("\nOptimized IR")
print("------------")
print(fused.graph)
print("\nPerformance comparison")
print("----------------------")
print(f"{'Metric':<22}{'Unfused':>12}{'Fused':>12}{'Reduction':>12}")
for name, before, after in (
    ("Instructions", len(unfused.program.instructions), len(fused.program.instructions)),
    ("Cycles", unfused.last_report.total_cycles, fused.last_report.total_cycles),
    (
        "DRAM bytes read",
        unfused.last_report.dram_bytes_read,
        fused.last_report.dram_bytes_read,
    ),
    (
        "DRAM bytes written",
        unfused.last_report.dram_bytes_written,
        fused.last_report.dram_bytes_written,
    ),
):
    reduction = 0.0 if before == 0 else 100.0 * (before - after) / before
    print(f"{name:<22}{before:>12}{after:>12}{reduction:>11.1f}%")

print("\nOptimized graph as DOT")
print("----------------------")
print(fused.graph.to_dot())

