import unittest

import numpy as np

import tinyaccel
from tinyaccel.isa import Opcode


def build_nchw_conv2d(
    input_shape=(1, 2, 5, 7),
    weight_shape=(3, 2, 3, 2),
    *,
    stride=(1, 1),
    padding=(1, 1, 0, 1),
    dilation=(1, 1),
) -> tinyaccel.Graph:
    builder = tinyaccel.GraphBuilder()
    input_value = builder.input("input", input_shape, layout="NCHW")
    weight = builder.input("weight", weight_shape, layout="OIHW")
    output = builder.conv2d(
        input_value,
        weight,
        stride=stride,
        padding=padding,
        dilation=dilation,
        name="output",
    )
    return builder.build(output)


class NchwConv2dSemanticTests(unittest.TestCase):
    def test_infers_nchw_shape_and_round_trips_ir(self) -> None:
        graph = build_nchw_conv2d(
            input_shape=(2, 3, 8, 11),
            weight_shape=(5, 3, 3, 2),
            stride=(2, 3),
            padding=(1, 2, 3, 0),
            dilation=(2, 1),
        )

        self.assertEqual(graph.outputs[0].type.shape, (2, 5, 4, 5))
        self.assertEqual(graph.outputs[0].type.layout, "NCHW")
        self.assertEqual(str(tinyaccel.parse_graph(str(graph))), str(graph))

    def test_reference_matches_manually_canonicalized_conv2d(self) -> None:
        graph = build_nchw_conv2d(
            stride=(2, 1),
            padding=(1, 2, 0, 1),
            dilation=(1, 2),
        )
        rng = np.random.default_rng(31)
        input_data = rng.standard_normal((1, 2, 5, 7), dtype=np.float32)
        weight_data = rng.standard_normal((3, 2, 3, 2), dtype=np.float32)

        actual = tinyaccel.evaluate(graph, input_data, weight_data)
        canonical = tinyaccel.conv2d_nhwc(
            np.transpose(input_data, (0, 2, 3, 1)),
            np.transpose(weight_data, (2, 3, 1, 0)),
            stride=(2, 1),
            padding=(1, 2, 0, 1),
            dilation=(1, 2),
        )

        np.testing.assert_allclose(
            actual,
            np.transpose(canonical, (0, 3, 1, 2)),
            rtol=1e-5,
            atol=1e-5,
        )


class Conv2dLayoutPassTests(unittest.TestCase):
    def test_canonicalization_expands_nchw_conv2d(self) -> None:
        graph = build_nchw_conv2d()

        canonical = tinyaccel.CanonicalizeConv2dLayoutsPass().run(graph)

        self.assertEqual(
            [operation.op for operation in canonical.operations],
            [
                "layout_transform",
                "layout_transform",
                "conv2d",
                "layout_transform",
            ],
        )
        conv2d = canonical.operations[2]
        self.assertEqual(conv2d.inputs[0].type.layout, "NHWC")
        self.assertEqual(conv2d.inputs[1].type.layout, "HWIO")
        self.assertEqual(conv2d.output.type.layout, "NHWC")
        self.assertEqual(canonical.outputs[0], graph.outputs[0])
        self.assertEqual(canonical.outputs[0].type.layout, "NCHW")

    def test_canonicalization_avoids_generated_name_collisions(self) -> None:
        builder = tinyaccel.GraphBuilder()
        input_value = builder.input(
            "output_input_nhwc", (1, 2, 5, 7), layout="NCHW"
        )
        weight = builder.input(
            "output_weight_hwio", (3, 2, 3, 2), layout="OIHW"
        )
        builder.input("output_nhwc", (1, 1, 1, 1), layout="NCHW")
        graph = builder.build(
            builder.conv2d(input_value, weight, name="output")
        )

        canonical = tinyaccel.CanonicalizeConv2dLayoutsPass().run(graph)

        names = [value.name for value in canonical.values]
        self.assertEqual(len(names), len(set(names)))
        self.assertIn("output_input_nhwc_1", names)
        self.assertIn("output_weight_hwio_1", names)
        self.assertIn("output_nhwc_1", names)

    def test_default_pipeline_eliminates_inverse_transform_pairs(self) -> None:
        builder = tinyaccel.GraphBuilder()
        input_nhwc = builder.input("input", (1, 5, 7, 2), layout="NHWC")
        weight_hwio = builder.input("weight", (3, 2, 2, 3), layout="HWIO")
        input_nchw = builder.layout_transform(input_nhwc, "NCHW")
        weight_oihw = builder.layout_transform(weight_hwio, "OIHW")
        output_nchw = builder.conv2d(
            input_nchw,
            weight_oihw,
            padding=(1, 1, 0, 1),
        )
        output_nhwc = builder.layout_transform(output_nchw, "NHWC")
        graph = builder.build(output_nhwc)

        optimized = tinyaccel.default_pipeline().run(graph)

        self.assertEqual(
            [operation.op for operation in optimized.operations], ["conv2d"]
        )
        conv2d = optimized.operations[0]
        self.assertEqual(conv2d.inputs, (input_nhwc, weight_hwio))
        self.assertEqual(optimized.outputs, (conv2d.output,))
        self.assertEqual(conv2d.output.type.layout, "NHWC")

    def test_default_pipeline_removes_simple_layout_round_trip(self) -> None:
        builder = tinyaccel.GraphBuilder()
        source = builder.input("source", (1, 2, 3, 4), layout="NCHW")
        nhwc = builder.layout_transform(source, "NHWC")
        restored = builder.layout_transform(nhwc, "NCHW")
        graph = builder.build(restored)

        optimized = tinyaccel.default_pipeline().run(graph)

        self.assertEqual(optimized.operations, ())
        self.assertEqual(optimized.outputs, (source,))

    def test_canonicalization_enables_constant_weight_transform_folding(self) -> None:
        builder = tinyaccel.GraphBuilder()
        input_value = builder.input("input", (1, 2, 5, 7), layout="NCHW")
        weight_data = np.arange(3 * 2 * 3 * 2, dtype=np.float32).reshape(
            3, 2, 3, 2
        )
        weight = builder.constant(weight_data, layout="OIHW", name="weight")
        graph = builder.build(
            builder.conv2d(
                input_value,
                weight,
                padding=(1, 1, 0, 1),
                name="output",
            )
        )

        optimized = tinyaccel.default_pipeline().run(graph)
        conv2d = next(
            operation for operation in optimized.operations if operation.op == "conv2d"
        )
        folded_weight = optimized.producer(conv2d.inputs[1])

        self.assertIsNotNone(folded_weight)
        self.assertEqual(folded_weight.op, "constant")
        self.assertEqual(folded_weight.output.type.layout, "HWIO")
        np.testing.assert_array_equal(
            folded_weight.attributes["value"],
            np.transpose(weight_data, (2, 3, 1, 0)),
        )


class NchwConv2dBackendTests(unittest.TestCase):
    def test_default_compile_matches_reference(self) -> None:
        graph = build_nchw_conv2d()
        executable = tinyaccel.compile(
            graph,
            options=tinyaccel.CompileOptions(
                tile_h=2,
                tile_w=3,
                tile_oc=2,
                tile_ic=1,
            ),
        )
        rng = np.random.default_rng(37)
        input_data = rng.standard_normal((1, 2, 5, 7), dtype=np.float32)
        weight_data = rng.standard_normal((3, 2, 3, 2), dtype=np.float32)

        actual = executable.run(input_data, weight_data)
        expected = tinyaccel.evaluate(graph, input_data, weight_data)

        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
        canonical_conv2d = next(
            operation
            for operation in executable.graph.operations
            if operation.op == "conv2d"
        )
        self.assertEqual(canonical_conv2d.inputs[0].type.layout, "NHWC")
        self.assertEqual(canonical_conv2d.inputs[1].type.layout, "HWIO")
        self.assertEqual(executable.program.output_shape, expected.shape)
        opcodes = [
            instruction.opcode for instruction in executable.program.instructions
        ]
        self.assertIn(Opcode.TRANSPOSE, opcodes)
        self.assertIn(Opcode.CONV2D, opcodes)

    def test_unoptimized_nchw_compile_has_clear_schedule_error(self) -> None:
        graph = build_nchw_conv2d()

        with self.assertRaisesRegex(ValueError, "optimization enabled"):
            tinyaccel.compile(
                graph,
                options=tinyaccel.CompileOptions(optimize=False),
            )


if __name__ == "__main__":
    unittest.main()
