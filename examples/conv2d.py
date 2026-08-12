"""Compile and execute a tiled NHWC Conv2D program."""

import numpy as np

import tinyaccel


def main() -> None:
    builder = tinyaccel.GraphBuilder()
    input_value = builder.input("input", (1, 6, 8, 2), layout="NHWC")
    weight = builder.input("weight", (3, 2, 2, 3), layout="HWIO")
    graph = builder.build(
        builder.conv2d(
            input_value,
            weight,
            stride=(2, 1),
            padding=(1, 1, 0, 1),
            name="output",
        )
    )
    executable = tinyaccel.compile(
        graph,
        options=tinyaccel.CompileOptions(tile_h=2, tile_w=4, tile_oc=2),
    )

    print("Graph IR")
    print("--------")
    print(graph)
    print()
    print("Conv2D Schedule IR")
    print("------------------")
    print(executable.schedule)
    print()
    print(executable.memory_plan)
    print()
    print("TinyAccel ISA")
    print("-------------")
    print(executable.program)

    rng = np.random.default_rng(4)
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
