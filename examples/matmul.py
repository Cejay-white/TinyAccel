"""Compile and simulate a tiled matrix multiplication."""

import numpy as np

import tinyaccel


builder = tinyaccel.GraphBuilder()
a = builder.input("a", (48, 40))
b = builder.input("b", (40, 24))
c = builder.matmul(a, b)
graph = builder.build(c)

executable = tinyaccel.compile(
    graph,
    options=tinyaccel.CompileOptions(tile_m=16, tile_n=16, tile_k=8),
)

rng = np.random.default_rng(0)
a_data = rng.standard_normal((48, 40), dtype=np.float32)
b_data = rng.standard_normal((40, 24), dtype=np.float32)
result = executable.run(a_data, b_data)

np.testing.assert_allclose(result, a_data @ b_data, rtol=1e-5, atol=1e-5)

print("Graph IR")
print("--------")
print(graph)
print("\nTinyAccel ISA")
print("-------------")
print(executable.program)
print()
print(executable.report())

