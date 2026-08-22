"""Inspect the v0.3 schedule IR and lifetime-aware SRAM memory plan."""

import numpy as np

import tinyaccel


def main() -> None:
    builder = tinyaccel.GraphBuilder()
    lhs = builder.input("lhs", (13, 9))
    rhs = builder.input("rhs", (9, 7))
    product = builder.matmul(lhs, rhs, name="product")
    positive = builder.relu(product, name="positive")
    stable = builder.relu(positive, name="stable")
    graph = builder.build(builder.relu(stable, name="result"))

    executable = tinyaccel.compile(
        graph,
        options=tinyaccel.CompileOptions(
            tile_m=5,
            tile_n=4,
            tile_k=3,
            optimize=False,
        ),
    )

    print("Schedule IR")
    print("-----------")
    print(executable.schedule)
    print()
    print(executable.memory_plan)

    rng = np.random.default_rng(3)
    lhs_data = rng.standard_normal((13, 9), dtype=np.float32)
    rhs_data = rng.standard_normal((9, 7), dtype=np.float32)
    actual = executable.run(lhs_data, rhs_data)
    expected = np.maximum(lhs_data @ rhs_data, 0)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)

    first = executable.memory_plan.allocation("product")
    last = executable.memory_plan.allocation("stable")
    print()
    print(f"Reused SRAM address: product/stable -> {first.offset}/{last.offset}")
    print("Final result residency: DRAM")
    print(f"Measured peak SRAM bytes: {executable.last_report.peak_sram_bytes}")


if __name__ == "__main__":
    main()
