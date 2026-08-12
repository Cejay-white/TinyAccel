import unittest
from unittest.mock import patch

import numpy as np

import tinyaccel
from tinyaccel.isa import Opcode


def build_conv2d(
    input_shape=(1, 5, 7, 2),
    weight_shape=(3, 2, 2, 4),
    *,
    stride=(1, 1),
    padding=(0, 0, 0, 0),
    dilation=(1, 1),
):
    builder = tinyaccel.GraphBuilder()
    input_value = builder.input("input", input_shape, layout="NHWC")
    weight = builder.input("weight", weight_shape, layout="HWIO")
    output = builder.conv2d(
        input_value,
        weight,
        stride=stride,
        padding=padding,
        dilation=dilation,
        name="output",
    )
    return builder.build(output)


class LayoutAndConvIrTests(unittest.TestCase):
    def test_layout_is_part_of_tensor_type(self) -> None:
        tensor_type = tinyaccel.TensorType((1, 4, 5, 3), "float32", "nhwc")

        self.assertEqual(tensor_type.layout, "NHWC")
        self.assertEqual(
            str(tensor_type), "tensor<1x4x5x3xfloat32, layout=NHWC>"
        )

    def test_layout_requires_rank_four(self) -> None:
        with self.assertRaisesRegex(ValueError, "rank-4"):
            tinyaccel.TensorType((3, 4), "float32", "NHWC")

    def test_conv2d_infers_shape_for_stride_padding_and_dilation(self) -> None:
        graph = build_conv2d(
            input_shape=(2, 8, 11, 3),
            weight_shape=(3, 2, 3, 5),
            stride=(2, 3),
            padding=(1, 2, 3, 0),
            dilation=(2, 1),
        )

        self.assertEqual(graph.outputs[0].type.shape, (2, 4, 5, 5))
        self.assertEqual(graph.outputs[0].type.layout, "NHWC")

    def test_conv2d_requires_nhwc_and_hwio(self) -> None:
        builder = tinyaccel.GraphBuilder()
        input_value = builder.input("input", (1, 5, 5, 2), layout="NCHW")
        weight = builder.input("weight", (3, 3, 2, 4), layout="HWIO")

        with self.assertRaisesRegex(ValueError, "NHWC input and HWIO weight"):
            builder.conv2d(input_value, weight)

    def test_conv2d_ir_round_trips_layout_and_attributes(self) -> None:
        graph = build_conv2d(stride=(2, 1), padding=(1, 2, 0, 1), dilation=(1, 2))

        restored = tinyaccel.parse_graph(str(graph))

        self.assertEqual(str(restored), str(graph))
        self.assertIn("layout=NHWC", str(graph))
        self.assertIn('"padding":[1,2,0,1]', str(graph))


class Conv2dReferenceTests(unittest.TestCase):
    def test_reference_handles_non_square_input_and_boundary_padding(self) -> None:
        graph = build_conv2d(
            input_shape=(1, 3, 5, 1),
            weight_shape=(2, 3, 1, 2),
            stride=(2, 1),
            padding=(1, 2, 2, 0),
        )
        input_data = np.arange(15, dtype=np.float32).reshape(1, 3, 5, 1)
        weight_data = np.arange(12, dtype=np.float32).reshape(2, 3, 1, 2) - 3

        actual = tinyaccel.evaluate(graph, input_data, weight_data)
        expected = _naive_conv2d(
            input_data,
            weight_data,
            stride=(2, 1),
            padding=(1, 2, 2, 0),
        )

        np.testing.assert_array_equal(actual, expected)

    def test_reference_handles_dilation(self) -> None:
        graph = build_conv2d(
            input_shape=(1, 7, 6, 2),
            weight_shape=(3, 2, 2, 3),
            padding=2,
            dilation=(2, 2),
        )
        rng = np.random.default_rng(5)
        input_data = rng.standard_normal((1, 7, 6, 2), dtype=np.float32)
        weight_data = rng.standard_normal((3, 2, 2, 3), dtype=np.float32)

        actual = tinyaccel.evaluate(graph, input_data, weight_data)
        expected = _naive_conv2d(
            input_data, weight_data, padding=(2, 2, 2, 2), dilation=(2, 2)
        )

        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


class Conv2dBackendTests(unittest.TestCase):
    def test_schedule_contains_spatial_and_reduction_axes(self) -> None:
        graph = build_conv2d()
        schedule = tinyaccel.create_schedule(
            graph, tile_h=2, tile_w=3, tile_oc=2
        )
        scheduled = schedule.operations[0]

        self.assertEqual(scheduled.output_tile_shape, (1, 2, 3, 2))
        self.assertEqual(
            tuple(loop.axis for loop in scheduled.loops),
            ("n", "h", "w", "oc", "kh", "kw", "ic"),
        )
        self.assertTrue(
            all(
                scheduled.loop(axis).kind == "reduction"
                for axis in ("kh", "kw", "ic")
            )
        )

    def test_tiled_isa_matches_reference_with_halo_padding(self) -> None:
        graph = build_conv2d(
            stride=(2, 1), padding=(1, 2, 0, 1), dilation=(1, 2)
        )
        executable = tinyaccel.compile(
            graph,
            options=tinyaccel.CompileOptions(
                tile_h=2, tile_w=3, tile_oc=2, optimize=False
            ),
        )
        rng = np.random.default_rng(9)
        input_data = rng.standard_normal((1, 5, 7, 2), dtype=np.float32)
        weight_data = rng.standard_normal((3, 2, 2, 4), dtype=np.float32)

        actual = executable.run(input_data, weight_data)
        expected = tinyaccel.evaluate(graph, input_data, weight_data)

        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
        opcodes = [instruction.opcode for instruction in executable.program.instructions]
        tile_count = (
            executable.schedule.operations[0].loop("n").tiles
            * executable.schedule.operations[0].loop("h").tiles
            * executable.schedule.operations[0].loop("w").tiles
            * executable.schedule.operations[0].loop("oc").tiles
        )
        self.assertEqual(opcodes.count(Opcode.CONV2D), tile_count)
        self.assertEqual(opcodes.count(Opcode.DMA_LOAD), 2 * tile_count)
        self.assertEqual(executable.last_report.dram_bytes_written, expected.nbytes)
        self.assertLessEqual(
            executable.last_report.peak_sram_bytes,
            executable.hardware.sram_bytes,
        )

    def test_lowering_obeys_injected_conv2d_schedule(self) -> None:
        graph = build_conv2d()
        operation = graph.operations[0]
        custom_schedule = tinyaccel.Schedule(
            graph,
            (
                tinyaccel.ScheduledOperation(
                    operation,
                    (
                        tinyaccel.LoopSpec("n", 1, 1),
                        tinyaccel.LoopSpec("h", 3, 1),
                        tinyaccel.LoopSpec("w", 6, 2),
                        tinyaccel.LoopSpec("oc", 4, 4),
                        tinyaccel.LoopSpec("kh", 3, 3, "reduction"),
                        tinyaccel.LoopSpec("kw", 2, 2, "reduction"),
                        tinyaccel.LoopSpec("ic", 2, 2, "reduction"),
                    ),
                ),
            ),
        )

        with patch(
            "tinyaccel.compiler.create_schedule", return_value=custom_schedule
        ):
            executable = tinyaccel.compile(
                graph,
                options=tinyaccel.CompileOptions(
                    tile_h=3, tile_w=6, tile_oc=1, optimize=False
                ),
            )

        opcodes = [instruction.opcode for instruction in executable.program.instructions]
        self.assertEqual(opcodes.count(Opcode.CONV2D), 9)

    def test_padding_halo_does_not_count_as_dram_traffic(self) -> None:
        graph = build_conv2d(
            input_shape=(1, 2, 2, 1),
            weight_shape=(3, 3, 1, 1),
            padding=1,
        )
        executable = tinyaccel.compile(
            graph,
            options=tinyaccel.CompileOptions(tile_h=2, tile_w=2, tile_oc=1),
        )
        input_data = np.ones((1, 2, 2, 1), dtype=np.float32)
        weight_data = np.ones((3, 3, 1, 1), dtype=np.float32)

        executable.run(input_data, weight_data)

        # Only four real input elements and nine weights cross DMA; the
        # twelve zero-valued halo elements are generated locally.
        self.assertEqual(executable.last_report.dram_bytes_read, (4 + 9) * 4)

    def test_conv2d_rejects_plan_plus_tile_sram_overflow(self) -> None:
        graph = build_conv2d(
            input_shape=(1, 4, 4, 1), weight_shape=(3, 3, 1, 2), padding=1
        )

        with self.assertRaisesRegex(ValueError, "conv2d tile requires"):
            tinyaccel.compile(
                graph,
                options=tinyaccel.CompileOptions(
                    tile_h=4, tile_w=4, tile_oc=2
                ),
                hardware=tinyaccel.HardwareConfig(sram_bytes=400),
            )


def _naive_conv2d(
    input_value,
    weight,
    *,
    stride=(1, 1),
    padding=(0, 0, 0, 0),
    dilation=(1, 1),
):
    n_size, input_h, input_w, input_c = input_value.shape
    kernel_h, kernel_w, _, output_c = weight.shape
    pad_top, pad_bottom, pad_left, pad_right = padding
    effective_h = (kernel_h - 1) * dilation[0] + 1
    effective_w = (kernel_w - 1) * dilation[1] + 1
    output_h = (input_h + pad_top + pad_bottom - effective_h) // stride[0] + 1
    output_w = (input_w + pad_left + pad_right - effective_w) // stride[1] + 1
    output = np.zeros((n_size, output_h, output_w, output_c), dtype=np.float32)
    for n_index in range(n_size):
        for output_y in range(output_h):
            for output_x in range(output_w):
                for output_channel in range(output_c):
                    for kernel_y in range(kernel_h):
                        input_y = output_y * stride[0] + kernel_y * dilation[0] - pad_top
                        if input_y < 0 or input_y >= input_h:
                            continue
                        for kernel_x in range(kernel_w):
                            input_x = output_x * stride[1] + kernel_x * dilation[1] - pad_left
                            if input_x < 0 or input_x >= input_w:
                                continue
                            for input_channel in range(input_c):
                                output[n_index, output_y, output_x, output_channel] += (
                                    input_value[n_index, input_y, input_x, input_channel]
                                    * weight[kernel_y, kernel_x, input_channel, output_channel]
                                )
    return output


if __name__ == "__main__":
    unittest.main()
