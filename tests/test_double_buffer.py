import unittest

import numpy as np

import tinyaccel
from tinyaccel.isa import Opcode


def build_matmul(m: int, k: int, n: int):
    builder = tinyaccel.GraphBuilder()
    lhs = builder.input("lhs", (m, k))
    rhs = builder.input("rhs", (k, n))
    return builder.build(builder.matmul(lhs, rhs, name="result"))


def build_conv2d():
    builder = tinyaccel.GraphBuilder()
    image = builder.input("image", (1, 5, 5, 4), layout="NHWC")
    weight = builder.input("weight", (3, 3, 4, 3), layout="HWIO")
    return builder.build(
        builder.conv2d(image, weight, padding=1, name="result")
    )


class DoubleBufferTests(unittest.TestCase):
    def test_compile_options_require_boolean_double_buffer_switch(self) -> None:
        with self.assertRaisesRegex(TypeError, "double_buffer must be a bool"):
            tinyaccel.CompileOptions(double_buffer=1)  # type: ignore[arg-type]

    def test_matmul_ping_pong_prefetch_overlaps_dma_and_compute(self) -> None:
        graph = build_matmul(4, 12, 4)
        hardware = tinyaccel.HardwareConfig(
            dma_bytes_per_cycle=4,
            macs_per_cycle=1,
            overlap_resources=True,
        )
        common = dict(tile_m=4, tile_n=4, tile_k=4)
        single = tinyaccel.compile(
            graph,
            options=tinyaccel.CompileOptions(**common),
            hardware=hardware,
        )
        double = tinyaccel.compile(
            graph,
            options=tinyaccel.CompileOptions(**common, double_buffer=True),
            hardware=hardware,
        )
        rng = np.random.default_rng(47)
        lhs = rng.standard_normal((4, 12), dtype=np.float32)
        rhs = rng.standard_normal((12, 4), dtype=np.float32)

        single_result = single.run(lhs, rhs)
        double_result = double.run(lhs, rhs)

        np.testing.assert_allclose(single_result, lhs @ rhs, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(double_result, lhs @ rhs, rtol=1e-5, atol=1e-5)
        self.assertEqual(single.last_report.sequential_cycles, 320)
        self.assertEqual(double.last_report.sequential_cycles, 320)
        self.assertEqual(single.last_report.total_cycles, 304)
        self.assertEqual(double.last_report.total_cycles, 240)
        self.assertEqual(single.last_report.peak_sram_bytes, 192)
        self.assertEqual(double.last_report.peak_sram_bytes, 320)

        buffers = [
            instruction.operands["buffer"]
            for instruction in double.program.instructions
            if instruction.opcode is Opcode.DMA_LOAD
        ]
        self.assertEqual(
            buffers,
            ["lhs_0", "rhs_0", "lhs_1", "rhs_1", "lhs_0", "rhs_0"],
        )
        first_matmul = double.last_report.timeline[3]
        second_tile_load = double.last_report.timeline[4]
        self.assertEqual(first_matmul.start_cycle, second_tile_load.start_cycle)
        self.assertLess(second_tile_load.end_cycle, first_matmul.end_cycle)

    def test_double_buffer_sram_check_and_single_tile_fallback(self) -> None:
        graph = build_matmul(4, 12, 4)
        hardware = tinyaccel.HardwareConfig(sram_bytes=319)

        tinyaccel.compile(
            graph,
            options=tinyaccel.CompileOptions(tile_m=4, tile_n=4, tile_k=4),
            hardware=hardware,
        )
        with self.assertRaisesRegex(ValueError, "matmul tile requires 320"):
            tinyaccel.compile(
                graph,
                options=tinyaccel.CompileOptions(
                    tile_m=4,
                    tile_n=4,
                    tile_k=4,
                    double_buffer=True,
                ),
                hardware=hardware,
            )

        one_tile = tinyaccel.compile(
            build_matmul(4, 4, 4),
            options=tinyaccel.CompileOptions(
                tile_m=4,
                tile_n=4,
                tile_k=4,
                double_buffer=True,
            ),
            hardware=tinyaccel.HardwareConfig(sram_bytes=192),
        )
        buffers = [
            instruction.operands["buffer"]
            for instruction in one_tile.program.instructions
            if instruction.opcode is Opcode.DMA_LOAD
        ]
        self.assertEqual(buffers, ["lhs", "rhs"])

        edge_reduction = tinyaccel.compile(
            build_matmul(4, 5, 4),
            options=tinyaccel.CompileOptions(
                tile_m=4,
                tile_n=4,
                tile_k=4,
                double_buffer=True,
            ),
            hardware=tinyaccel.HardwareConfig(sram_bytes=224),
        )
        values = np.ones((4, 5), dtype=np.float32)
        weights = np.ones((5, 4), dtype=np.float32)
        edge_reduction.run(values, weights)
        self.assertEqual(edge_reduction.last_report.peak_sram_bytes, 224)

    def test_fused_matmul_bias_relu_uses_ping_pong_buffers(self) -> None:
        builder = tinyaccel.GraphBuilder()
        lhs_value = builder.input("lhs", (4, 12))
        rhs_value = builder.input("rhs", (12, 4))
        bias_value = builder.input("bias", (4,))
        graph = builder.build(
            builder.relu(
                builder.add(builder.matmul(lhs_value, rhs_value), bias_value),
                name="result",
            )
        )
        executable = tinyaccel.compile(
            graph,
            options=tinyaccel.CompileOptions(
                tile_m=4,
                tile_n=4,
                tile_k=4,
                double_buffer=True,
            ),
            hardware=tinyaccel.HardwareConfig(overlap_resources=True),
        )
        rng = np.random.default_rng(53)
        feeds = (
            rng.standard_normal((4, 12), dtype=np.float32),
            rng.standard_normal((12, 4), dtype=np.float32),
            rng.standard_normal((4,), dtype=np.float32),
        )

        actual = executable.run(*feeds)
        expected = tinyaccel.evaluate(graph, *feeds)

        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
        self.assertEqual(executable.graph.operations[0].op, "matmul_bias_relu")
        load_buffers = [
            instruction.operands["buffer"]
            for instruction in executable.program.instructions
            if instruction.opcode is Opcode.DMA_LOAD
        ]
        self.assertEqual(load_buffers[-1], "rhs_0")
        self.assertIn("lhs_1", load_buffers)

    def test_direct_and_im2col_conv2d_pipeline_reduction_tiles(self) -> None:
        graph = build_conv2d()
        rng = np.random.default_rng(59)
        feeds = (
            rng.standard_normal((1, 5, 5, 4), dtype=np.float32),
            rng.standard_normal((3, 3, 4, 3), dtype=np.float32),
        )
        expected = tinyaccel.evaluate(graph, *feeds)
        hardware = tinyaccel.HardwareConfig(overlap_resources=True)

        for lowering in ("direct", "im2col"):
            with self.subTest(lowering=lowering):
                common = dict(
                    tile_h=3,
                    tile_w=3,
                    tile_ic=2,
                    tile_oc=3,
                    conv2d_lowering=lowering,
                )
                single = tinyaccel.compile(
                    graph,
                    options=tinyaccel.CompileOptions(**common),
                    hardware=hardware,
                )
                double = tinyaccel.compile(
                    graph,
                    options=tinyaccel.CompileOptions(
                        **common, double_buffer=True
                    ),
                    hardware=hardware,
                )

                single.run(*feeds)
                actual = double.run(*feeds)

                np.testing.assert_allclose(
                    actual, expected, rtol=1e-5, atol=1e-5
                )
                self.assertEqual(
                    single.last_report.sequential_cycles,
                    double.last_report.sequential_cycles,
                )
                self.assertLess(
                    double.last_report.total_cycles,
                    single.last_report.total_cycles,
                )
                self.assertGreater(
                    double.last_report.peak_sram_bytes,
                    single.last_report.peak_sram_bytes,
                )
                self.assertEqual(
                    double.last_report.dram_bytes_read,
                    single.last_report.dram_bytes_read,
                )
                load_buffers = [
                    instruction.operands["buffer"]
                    for instruction in double.program.instructions
                    if instruction.opcode is Opcode.DMA_LOAD
                ]
                self.assertIn("input_0", load_buffers)
                self.assertIn("input_1", load_buffers)
                self.assertIn("weight_0", load_buffers)
                self.assertIn("weight_1", load_buffers)


if __name__ == "__main__":
    unittest.main()
