import unittest

import numpy as np

import tinyaccel
from tinyaccel.isa import Opcode


LAYOUT_CASES = (
    ("NCHW", "NHWC", (2, 3, 5, 7), (0, 2, 3, 1)),
    ("NHWC", "NCHW", (2, 5, 7, 3), (0, 3, 1, 2)),
    ("OIHW", "HWIO", (11, 3, 2, 5), (2, 3, 1, 0)),
    ("HWIO", "OIHW", (2, 5, 3, 11), (3, 2, 0, 1)),
)


def build_layout_transform(
    source_layout: str,
    target_layout: str,
    shape: tuple[int, int, int, int],
) -> tinyaccel.Graph:
    builder = tinyaccel.GraphBuilder()
    source = builder.input("source", shape, layout=source_layout)
    output = builder.layout_transform(source, target_layout, name="output")
    return builder.build(output)


class LayoutTransformIrTests(unittest.TestCase):
    def test_infers_all_supported_layout_pairs(self) -> None:
        for source_layout, target_layout, shape, permutation in LAYOUT_CASES:
            with self.subTest(pair=f"{source_layout}->{target_layout}"):
                graph = build_layout_transform(source_layout, target_layout, shape)
                expected_shape = tuple(shape[axis] for axis in permutation)

                self.assertEqual(
                    tinyaccel.layout_permutation(source_layout, target_layout),
                    permutation,
                )
                self.assertEqual(graph.outputs[0].type.shape, expected_shape)
                self.assertEqual(graph.outputs[0].type.layout, target_layout)
                self.assertEqual(
                    graph.operations[0].attributes,
                    {"target_layout": target_layout},
                )

    def test_round_trips_canonical_ir(self) -> None:
        graph = build_layout_transform("NCHW", "NHWC", (1, 3, 5, 7))

        restored = tinyaccel.parse_graph(str(graph))

        self.assertEqual(str(restored), str(graph))
        self.assertIn('"target_layout":"NHWC"', str(graph))

    def test_rejects_missing_unsupported_and_identity_layouts(self) -> None:
        builder = tinyaccel.GraphBuilder()
        untyped = builder.input("untyped", (1, 2, 3, 4))
        with self.assertRaisesRegex(ValueError, "source layout"):
            builder.layout_transform(untyped, "NHWC")

        other_builder = tinyaccel.GraphBuilder()
        activation = other_builder.input(
            "activation", (1, 2, 3, 4), layout="NCHW"
        )
        for target in ("HWIO", "NCHW", "CHWN"):
            with self.subTest(target=target):
                with self.assertRaisesRegex(ValueError, "unsupported layout transform"):
                    other_builder.layout_transform(activation, target)


class LayoutTransformSemanticTests(unittest.TestCase):
    def test_reference_matches_numpy_for_all_layout_pairs(self) -> None:
        for source_layout, target_layout, shape, permutation in LAYOUT_CASES:
            with self.subTest(pair=f"{source_layout}->{target_layout}"):
                graph = build_layout_transform(source_layout, target_layout, shape)
                source = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)

                actual = tinyaccel.evaluate(graph, source)

                np.testing.assert_array_equal(
                    actual, np.transpose(source, permutation)
                )

    def test_constant_folding_preserves_transformed_layout(self) -> None:
        source = np.arange(2 * 3 * 4 * 5, dtype=np.float32).reshape(2, 3, 4, 5)
        builder = tinyaccel.GraphBuilder()
        constant = builder.constant(source, layout="NCHW", name="source")
        output = builder.layout_transform(constant, "NHWC", name="output")
        graph = builder.build(output)

        optimized = tinyaccel.default_pipeline().run(graph)

        self.assertEqual(len(optimized.operations), 1)
        self.assertEqual(optimized.operations[0].op, "constant")
        self.assertEqual(optimized.outputs[0].type.layout, "NHWC")
        np.testing.assert_array_equal(
            optimized.operations[0].attributes["value"],
            np.transpose(source, (0, 2, 3, 1)),
        )


class LayoutTransformBackendTests(unittest.TestCase):
    def test_schedule_uses_layout_axes_and_requested_tiles(self) -> None:
        graph = build_layout_transform("NCHW", "NHWC", (2, 3, 5, 7))

        scheduled = tinyaccel.create_schedule(
            graph, tile_h=2, tile_w=3, tile_ic=2
        ).operations[0]

        self.assertEqual(
            tuple(loop.axis for loop in scheduled.loops),
            ("n", "h", "w", "c"),
        )
        self.assertEqual(scheduled.output_tile_shape, (1, 2, 3, 2))

    def test_tiled_isa_matches_numpy_for_all_layout_pairs(self) -> None:
        expected_axes = {
            "NCHW": ("n", "c", "h", "w"),
            "NHWC": ("n", "h", "w", "c"),
            "OIHW": ("oc", "ic", "kh", "kw"),
            "HWIO": ("kh", "kw", "ic", "oc"),
        }
        for source_layout, target_layout, shape, permutation in LAYOUT_CASES:
            with self.subTest(pair=f"{source_layout}->{target_layout}"):
                graph = build_layout_transform(source_layout, target_layout, shape)
                executable = tinyaccel.compile(
                    graph,
                    options=tinyaccel.CompileOptions(
                        tile_h=2,
                        tile_w=3,
                        tile_oc=4,
                        tile_ic=2,
                        optimize=False,
                    ),
                )
                source = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)

                actual = executable.run(source)
                expected = np.transpose(source, permutation)

                np.testing.assert_array_equal(actual, expected)
                scheduled = executable.schedule.operations[0]
                self.assertEqual(
                    tuple(loop.axis for loop in scheduled.loops),
                    expected_axes[target_layout],
                )
                tile_count = int(
                    np.prod([loop.tiles for loop in scheduled.loops])
                )
                opcodes = [
                    instruction.opcode
                    for instruction in executable.program.instructions
                ]
                self.assertEqual(opcodes.count(Opcode.DMA_LOAD), tile_count)
                self.assertEqual(opcodes.count(Opcode.TRANSPOSE), tile_count)
                self.assertEqual(opcodes.count(Opcode.DMA_STORE), tile_count)
                self.assertEqual(executable.last_report.dram_bytes_read, source.nbytes)
                self.assertEqual(
                    executable.last_report.dram_bytes_written, expected.nbytes
                )

    def test_transforms_nchw_conv2d_inputs_through_planned_sram(self) -> None:
        builder = tinyaccel.GraphBuilder()
        source = builder.input("source", (1, 2, 5, 7), layout="NCHW")
        weight = builder.input("weight", (3, 2, 3, 2), layout="OIHW")
        nhwc = builder.layout_transform(source, "NHWC", name="nhwc")
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
                tile_h=2,
                tile_w=3,
                tile_oc=2,
                tile_ic=1,
                optimize=False,
            ),
        )
        rng = np.random.default_rng(27)
        source_data = rng.standard_normal(source.type.shape, dtype=np.float32)
        weight_data = rng.standard_normal(weight.type.shape, dtype=np.float32)

        actual = executable.run(source_data, weight_data)
        expected = tinyaccel.evaluate(graph, source_data, weight_data)

        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
        self.assertIs(
            executable.program.value_spaces[nhwc.name], tinyaccel.MemorySpace.SRAM
        )
        self.assertIs(
            executable.program.value_spaces[hwio.name], tinyaccel.MemorySpace.SRAM
        )
        self.assertGreater(executable.last_report.sram_bytes_read, 0)
        self.assertGreater(executable.last_report.sram_bytes_written, 0)

    def test_rejects_layout_tile_that_exceeds_sram(self) -> None:
        graph = build_layout_transform("NCHW", "NHWC", (1, 4, 4, 4))

        with self.assertRaisesRegex(ValueError, "layout_transform tile"):
            tinyaccel.compile(
                graph,
                options=tinyaccel.CompileOptions(optimize=False),
                hardware=tinyaccel.HardwareConfig(sram_bytes=511),
            )


if __name__ == "__main__":
    unittest.main()
