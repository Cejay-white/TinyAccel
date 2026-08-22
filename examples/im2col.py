"""Compare direct and im2col lowering for the same Conv2D schedule."""

from collections import Counter

import numpy as np

import tinyaccel


def main() -> None:
    builder = tinyaccel.GraphBuilder()
    input_value = builder.input("input", (1, 8, 8, 4), layout="NHWC")
    weight = builder.input("weight", (3, 3, 4, 8), layout="HWIO")
    graph = builder.build(
        builder.conv2d(input_value, weight, padding=1, name="output")
    )
    common_options = dict(
        tile_h=4,
        tile_w=4,
        tile_oc=4,
        tile_ic=2,
        optimize=False,
    )
    direct = tinyaccel.compile(
        graph,
        options=tinyaccel.CompileOptions(
            **common_options, conv2d_lowering="direct"
        ),
    )
    im2col = tinyaccel.compile(
        graph,
        options=tinyaccel.CompileOptions(
            **common_options, conv2d_lowering="im2col"
        ),
    )

    rng = np.random.default_rng(12)
    input_data = rng.standard_normal(input_value.type.shape, dtype=np.float32)
    weight_data = rng.standard_normal(weight.type.shape, dtype=np.float32)
    expected = tinyaccel.evaluate(graph, input_data, weight_data)
    direct_result = direct.run(input_data, weight_data)
    im2col_result = im2col.run(input_data, weight_data)
    np.testing.assert_allclose(direct_result, expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(im2col_result, expected, rtol=1e-5, atol=1e-5)

    print("Shared Conv2D Schedule")
    print("----------------------")
    print(direct.schedule)
    print()
    print(f"{'Metric':<24} {'Direct':>12} {'Im2col':>12}")
    print("-" * 50)
    metrics = (
        (
            "Instructions",
            len(direct.program.instructions),
            len(im2col.program.instructions),
        ),
        (
            "Total cycles",
            direct.last_report.total_cycles,
            im2col.last_report.total_cycles,
        ),
        (
            "Layout cycles",
            direct.last_report.layout_cycles,
            im2col.last_report.layout_cycles,
        ),
        (
            "DRAM bytes read",
            direct.last_report.dram_bytes_read,
            im2col.last_report.dram_bytes_read,
        ),
        (
            "DRAM bytes written",
            direct.last_report.dram_bytes_written,
            im2col.last_report.dram_bytes_written,
        ),
        (
            "Peak SRAM bytes",
            direct.last_report.peak_sram_bytes,
            im2col.last_report.peak_sram_bytes,
        ),
    )
    for name, direct_value, im2col_value in metrics:
        print(f"{name:<24} {direct_value:>12} {im2col_value:>12}")

    print()
    for name, executable in (("Direct", direct), ("Im2col", im2col)):
        counts = Counter(
            instruction.opcode.value
            for instruction in executable.program.instructions
        )
        summary = ", ".join(
            f"{opcode}={count}" for opcode, count in sorted(counts.items())
        )
        print(f"{name:<7} opcodes: {summary}")
    print("\nNumPy correctness: PASS")


if __name__ == "__main__":
    main()
