"""Compare sequential and resource-overlapped analytical timing."""

import numpy as np

import tinyaccel


builder = tinyaccel.GraphBuilder()
image = builder.input("image", (1, 8, 8, 4), layout="NHWC")
weight = builder.input("weight", (3, 3, 4, 8), layout="HWIO")
graph = builder.build(
    builder.conv2d(image, weight, padding=(1, 1, 1, 1), name="result")
)
options = tinyaccel.CompileOptions(
    tile_h=4,
    tile_w=4,
    tile_ic=2,
    tile_oc=4,
    conv2d_lowering="im2col",
)


def compile_with_timing(*, overlap_resources: bool) -> tinyaccel.Executable:
    return tinyaccel.compile(
        graph,
        options=options,
        hardware=tinyaccel.HardwareConfig(
            overlap_resources=overlap_resources,
        ),
    )


rng = np.random.default_rng(41)
image_data = rng.standard_normal(image.type.shape, dtype=np.float32)
weight_data = rng.standard_normal(weight.type.shape, dtype=np.float32)
sequential = compile_with_timing(overlap_resources=False)
overlapped = compile_with_timing(overlap_resources=True)

sequential_result = sequential.run(image_data, weight_data)
overlapped_result = overlapped.run(image_data, weight_data)
expected = tinyaccel.evaluate(graph, image_data, weight_data)
np.testing.assert_allclose(sequential_result, expected, rtol=1e-5, atol=1e-5)
np.testing.assert_allclose(overlapped_result, expected, rtol=1e-5, atol=1e-5)

print("Resource timing comparison")
print("=" * 58)
print(f"{'Metric':<24} {'Sequential':>14} {'Overlapped':>14}")
print("-" * 58)
rows = (
    (
        "Sequential work",
        sequential.last_report.sequential_cycles,
        overlapped.last_report.sequential_cycles,
    ),
    (
        "Elapsed cycles",
        sequential.last_report.total_cycles,
        overlapped.last_report.total_cycles,
    ),
    (
        "Overlap saved",
        sequential.last_report.overlap_cycles_saved,
        overlapped.last_report.overlap_cycles_saved,
    ),
    (
        "DMA busy cycles",
        sequential.last_report.resource_cycles["DMA"],
        overlapped.last_report.resource_cycles["DMA"],
    ),
    (
        "Compute busy cycles",
        sequential.last_report.resource_cycles["COMPUTE"],
        overlapped.last_report.resource_cycles["COMPUTE"],
    ),
    (
        "Layout busy cycles",
        sequential.last_report.resource_cycles["LAYOUT"],
        overlapped.last_report.resource_cycles["LAYOUT"],
    ),
)
for label, serial_value, overlap_value in rows:
    print(f"{label:<24} {serial_value:>14} {overlap_value:>14}")

print()
print(overlapped.timeline(width=44, max_events=14))
