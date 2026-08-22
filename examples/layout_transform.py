"""Transform NCHW/OIHW tensors before running canonical NHWC Conv2D."""

import numpy as np

import tinyaccel


def main() -> None:
    builder = tinyaccel.GraphBuilder()
    input_value = builder.input("input", (1, 2, 5, 7), layout="NCHW")
    weight = builder.input("weight", (3, 2, 3, 2), layout="OIHW")
    nhwc = builder.layout_transform(input_value, "NHWC", name="nhwc")
    hwio = builder.layout_transform(weight, "HWIO", name="hwio")
    output = builder.conv2d(
        nhwc,
        hwio,
        padding=(1, 1, 0, 1),
        name="output",
    )
    graph = builder.build(output)
    executable = tinyaccel.compile(
        graph,
        options=tinyaccel.CompileOptions(
            tile_h=5,
            tile_w=7,
            tile_oc=3,
            tile_ic=2,
            optimize=False,
        ),
    )

    print("Graph IR")
    print("--------")
    print(graph)
    print()
    print("Schedule IR")
    print("-----------")
    print(executable.schedule)
    print()
    print(executable.memory_plan)
    print()
    print("TinyAccel ISA")
    print("-------------")
    print(executable.program)

    rng = np.random.default_rng(8)
    input_data = rng.standard_normal(input_value.type.shape, dtype=np.float32)
    weight_data = rng.standard_normal(weight.type.shape, dtype=np.float32)
    actual = executable.run(input_data, weight_data)
    expected = tinyaccel.evaluate(graph, input_data, weight_data)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)

    print()
    print(executable.report())
    print("\nNumPy correctness: PASS")


if __name__ == "__main__":
    main()
