import unittest

import numpy as np

import tinyaccel
from tinyaccel.isa import Opcode


def build_conv2d(
    input_shape=(1, 5, 7, 3),
    weight_shape=(3, 2, 3, 4),
    *,
    input_layout="NHWC",
    weight_layout="HWIO",
    stride=(1, 1),
    padding=(0, 0, 0, 0),
    dilation=(1, 1),
) -> tinyaccel.Graph:
    builder = tinyaccel.GraphBuilder()
    input_value = builder.input("input", input_shape, layout=input_layout)
    weight = builder.input("weight", weight_shape, layout=weight_layout)
    output = builder.conv2d(
        input_value,
        weight,
        stride=stride,
        padding=padding,
        dilation=dilation,
        name="output",
    )
    return builder.build(output)


class LayoutCostModelTests(unittest.TestCase):
    def test_hardware_validates_vector_throughput(self) -> None:
        with self.assertRaisesRegex(ValueError, "vector_elements_per_cycle"):
            tinyaccel.HardwareConfig(vector_elements_per_cycle=0)

    def test_transpose_cycles_use_vector_throughput(self) -> None:
        builder = tinyaccel.GraphBuilder()
        source = builder.input("source", (1, 2, 3, 4), layout="NCHW")
        graph = builder.build(
            builder.layout_transform(source, "NHWC", name="output")
        )
        executable = tinyaccel.compile(
            graph,
            options=tinyaccel.CompileOptions(
                tile_h=3,
                tile_w=4,
                tile_ic=2,
                optimize=False,
            ),
            hardware=tinyaccel.HardwareConfig(vector_elements_per_cycle=5),
        )
        data = np.arange(24, dtype=np.float32).reshape(1, 2, 3, 4)

        executable.run(data)

        self.assertEqual(executable.last_report.layout_cycles, 5)
        self.assertEqual(
            executable.last_report.cycles_by_opcode[Opcode.TRANSPOSE.value],
            5,
        )
        self.assertIn("Layout cycles:      5", executable.report())


class Im2colLoweringTests(unittest.TestCase):
    def test_compile_options_reject_unknown_conv2d_lowering(self) -> None:
        with self.assertRaisesRegex(ValueError, "direct.*im2col"):
            tinyaccel.CompileOptions(conv2d_lowering="implicit_gemm")

    def test_im2col_matches_direct_and_exposes_layout_overhead(self) -> None:
        graph = build_conv2d(
            input_shape=(2, 5, 7, 3),
            stride=(2, 1),
            padding=(1, 2, 0, 1),
            dilation=(1, 2),
        )
        common_options = dict(
            tile_h=2,
            tile_w=4,
            tile_oc=3,
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
        rng = np.random.default_rng(41)
        input_data = rng.standard_normal((2, 5, 7, 3), dtype=np.float32)
        weight_data = rng.standard_normal((3, 2, 3, 4), dtype=np.float32)

        direct_result = direct.run(input_data, weight_data)
        im2col_result = im2col.run(input_data, weight_data)
        expected = tinyaccel.evaluate(graph, input_data, weight_data)

        np.testing.assert_allclose(direct_result, expected, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(im2col_result, expected, rtol=1e-5, atol=1e-5)
        direct_opcodes = [
            instruction.opcode for instruction in direct.program.instructions
        ]
        im2col_opcodes = [
            instruction.opcode for instruction in im2col.program.instructions
        ]
        self.assertIn(Opcode.CONV2D, direct_opcodes)
        self.assertNotIn(Opcode.IM2COL, direct_opcodes)
        self.assertNotIn(Opcode.CONV2D, im2col_opcodes)
        self.assertEqual(
            im2col_opcodes.count(Opcode.IM2COL),
            im2col_opcodes.count(Opcode.MATMUL),
        )
        self.assertGreater(im2col_opcodes.count(Opcode.RESHAPE), 0)
        self.assertEqual(direct.last_report.layout_cycles, 0)
        self.assertGreater(im2col.last_report.layout_cycles, 0)
        self.assertGreater(
            im2col.last_report.total_cycles, direct.last_report.total_cycles
        )
        self.assertGreater(
            im2col.last_report.peak_sram_bytes,
            direct.last_report.peak_sram_bytes,
        )
        self.assertEqual(
            im2col.last_report.dram_bytes_read,
            direct.last_report.dram_bytes_read,
        )
        self.assertEqual(
            im2col.last_report.dram_bytes_written,
            direct.last_report.dram_bytes_written,
        )

    def test_im2col_runs_after_nchw_canonicalization(self) -> None:
        graph = build_conv2d(
            input_shape=(1, 2, 5, 7),
            weight_shape=(3, 2, 3, 2),
            input_layout="NCHW",
            weight_layout="OIHW",
            padding=(1, 1, 0, 1),
        )
        executable = tinyaccel.compile(
            graph,
            options=tinyaccel.CompileOptions(
                tile_h=2,
                tile_w=3,
                tile_oc=2,
                tile_ic=1,
                conv2d_lowering="im2col",
            ),
        )
        rng = np.random.default_rng(43)
        input_data = rng.standard_normal((1, 2, 5, 7), dtype=np.float32)
        weight_data = rng.standard_normal((3, 2, 3, 2), dtype=np.float32)

        actual = executable.run(input_data, weight_data)
        expected = tinyaccel.evaluate(graph, input_data, weight_data)

        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
        opcodes = [
            instruction.opcode for instruction in executable.program.instructions
        ]
        self.assertIn(Opcode.TRANSPOSE, opcodes)
        self.assertIn(Opcode.IM2COL, opcodes)
        self.assertIn(Opcode.MATMUL, opcodes)
        self.assertNotIn(Opcode.CONV2D, opcodes)

    def test_im2col_rejects_columns_that_exceed_sram(self) -> None:
        graph = build_conv2d(
            input_shape=(1, 4, 4, 2),
            weight_shape=(3, 3, 2, 2),
        )
        hardware = tinyaccel.HardwareConfig(sram_bytes=400)

        tinyaccel.compile(
            graph,
            options=tinyaccel.CompileOptions(
                optimize=False, conv2d_lowering="direct"
            ),
            hardware=hardware,
        )
        with self.assertRaisesRegex(ValueError, "im2col conv2d tile"):
            tinyaccel.compile(
                graph,
                options=tinyaccel.CompileOptions(
                    optimize=False, conv2d_lowering="im2col"
                ),
                hardware=hardware,
            )


if __name__ == "__main__":
    unittest.main()
