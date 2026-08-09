import unittest

import numpy as np

import tinyaccel
from tinyaccel.isa import Opcode


def build_matmul(m: int, k: int, n: int):
    builder = tinyaccel.GraphBuilder()
    lhs = builder.input("lhs", (m, k))
    rhs = builder.input("rhs", (k, n))
    return builder.build(builder.matmul(lhs, rhs, name="result"))


class CompilerTests(unittest.TestCase):
    def test_lowers_matmul_to_tiled_isa(self) -> None:
        executable = tinyaccel.compile(
            build_matmul(5, 7, 3),
            options=tinyaccel.CompileOptions(tile_m=4, tile_n=2, tile_k=3),
        )
        opcodes = [instruction.opcode for instruction in executable.program.instructions]

        # 2 M tiles * 2 N tiles, with 3 K tiles per output tile.
        self.assertEqual(opcodes.count(Opcode.ZERO), 4)
        self.assertEqual(opcodes.count(Opcode.DMA_LOAD), 24)
        self.assertEqual(opcodes.count(Opcode.MATMUL), 12)
        self.assertEqual(opcodes.count(Opcode.DMA_STORE), 4)

    def test_rejects_tiles_larger_than_sram(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires .* SRAM bytes"):
            tinyaccel.compile(
                build_matmul(64, 64, 64),
                options=tinyaccel.CompileOptions(tile_m=64, tile_n=64, tile_k=64),
                hardware=tinyaccel.HardwareConfig(sram_bytes=1024),
            )

    def test_sram_check_uses_actual_edge_tile_size(self) -> None:
        executable = tinyaccel.compile(
            build_matmul(2, 2, 2),
            options=tinyaccel.CompileOptions(tile_m=64, tile_n=64, tile_k=64),
            hardware=tinyaccel.HardwareConfig(sram_bytes=48),
        )

        self.assertEqual(len(executable.program.instructions), 5)

    def test_optimization_removes_add_zero_before_lowering(self) -> None:
        builder = tinyaccel.GraphBuilder()
        value = builder.input("value", (4, 4))
        zero = builder.constant(0.0, dtype="float32", name="zero")
        graph = builder.build(builder.relu(builder.add(value, zero)))

        executable = tinyaccel.compile(graph)

        self.assertEqual([op.op for op in executable.graph.operations], ["relu"])
        self.assertNotIn(
            Opcode.ADD,
            [instruction.opcode for instruction in executable.program.instructions],
        )


class SimulatorTests(unittest.TestCase):
    def test_matches_numpy_for_edge_tiles(self) -> None:
        executable = tinyaccel.compile(
            build_matmul(35, 19, 21),
            options=tinyaccel.CompileOptions(tile_m=16, tile_n=8, tile_k=7),
        )
        rng = np.random.default_rng(7)
        lhs = rng.standard_normal((35, 19), dtype=np.float32)
        rhs = rng.standard_normal((19, 21), dtype=np.float32)

        actual = executable.run(lhs, rhs)

        np.testing.assert_allclose(actual, lhs @ rhs, rtol=1e-5, atol=1e-5)
        self.assertGreater(executable.last_report.total_cycles, 0)
        self.assertEqual(executable.last_report.dram_bytes_written, actual.nbytes)
        self.assertLessEqual(
            executable.last_report.peak_sram_bytes,
            executable.hardware.sram_bytes,
        )
        self.assertEqual(
            len(executable.last_report.timeline),
            len(executable.program.instructions),
        )
        self.assertEqual(executable.last_report.timeline[0].start_cycle, 0)
        self.assertEqual(
            executable.last_report.timeline[-1].end_cycle,
            executable.last_report.total_cycles,
        )

        timeline = executable.timeline(width=30, max_events=6)
        self.assertIn("Instruction Timeline", timeline)
        self.assertIn("instructions omitted", timeline)
        self.assertIn("MATMUL", timeline)

    def test_validates_input_shape_and_dtype(self) -> None:
        executable = tinyaccel.compile(build_matmul(2, 3, 4))

        with self.assertRaisesRegex(ValueError, "shape"):
            executable.run(
                np.ones((2, 2), dtype=np.float32),
                np.ones((3, 4), dtype=np.float32),
            )
        with self.assertRaisesRegex(ValueError, "dtype"):
            executable.run(
                np.ones((2, 3), dtype=np.float64),
                np.ones((3, 4), dtype=np.float32),
            )

    def test_executes_multi_operator_graph_with_bias_broadcast(self) -> None:
        builder = tinyaccel.GraphBuilder()
        lhs = builder.input("lhs", (17, 11))
        rhs = builder.input("rhs", (11, 13))
        bias = builder.constant(np.linspace(-1, 1, 13, dtype=np.float32), name="bias")
        product = builder.matmul(lhs, rhs, name="product")
        biased = builder.add(product, bias, name="biased")
        graph = builder.build(builder.relu(biased, name="result"))
        executable = tinyaccel.compile(
            graph,
            options=tinyaccel.CompileOptions(tile_m=8, tile_n=5, tile_k=4),
        )
        rng = np.random.default_rng(11)
        lhs_data = rng.standard_normal((17, 11), dtype=np.float32)
        rhs_data = rng.standard_normal((11, 13), dtype=np.float32)

        actual = executable.run(lhs_data, rhs_data)
        expected = tinyaccel.evaluate(graph, lhs_data, rhs_data)

        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
        opcodes = [instruction.opcode for instruction in executable.program.instructions]
        self.assertIn(Opcode.MATMUL, opcodes)
        self.assertIn(Opcode.ADD, opcodes)
        self.assertIn(Opcode.RELU, opcodes)
        self.assertGreater(executable.last_report.dram_bytes_read, 0)

    def test_executes_scalar_broadcast_add(self) -> None:
        builder = tinyaccel.GraphBuilder()
        value = builder.input("value", (3, 5))
        scalar = builder.constant(2.5, dtype="float32", name="scalar")
        graph = builder.build(builder.add(value, scalar, name="result"))
        executable = tinyaccel.compile(
            graph,
            options=tinyaccel.CompileOptions(tile_m=2, tile_n=3),
        )
        data = np.arange(15, dtype=np.float32).reshape(3, 5)

        np.testing.assert_array_equal(executable.run(data), data + 2.5)

    def test_fusion_reduces_dram_traffic_and_cycles(self) -> None:
        builder = tinyaccel.GraphBuilder()
        lhs = builder.input("lhs", (16, 12))
        rhs = builder.input("rhs", (12, 10))
        bias = builder.input("bias", (10,))
        product = builder.matmul(lhs, rhs)
        biased = builder.add(product, bias)
        graph = builder.build(builder.relu(biased, name="result"))
        options = dict(tile_m=8, tile_n=5, tile_k=4)
        fused = tinyaccel.compile(
            graph, options=tinyaccel.CompileOptions(**options, optimize=True)
        )
        unfused = tinyaccel.compile(
            graph, options=tinyaccel.CompileOptions(**options, optimize=False)
        )
        rng = np.random.default_rng(17)
        feeds = (
            rng.standard_normal((16, 12), dtype=np.float32),
            rng.standard_normal((12, 10), dtype=np.float32),
            rng.standard_normal((10,), dtype=np.float32),
        )

        fused_result = fused.run(*feeds)
        unfused_result = unfused.run(*feeds)

        np.testing.assert_allclose(fused_result, unfused_result, rtol=1e-5, atol=1e-5)
        self.assertEqual([op.op for op in fused.graph.operations], ["matmul_bias_relu"])
        self.assertLess(
            fused.last_report.dram_bytes_written,
            unfused.last_report.dram_bytes_written,
        )
        self.assertLess(fused.last_report.total_cycles, unfused.last_report.total_cycles)

    def test_executes_fully_constant_optimized_graph(self) -> None:
        builder = tinyaccel.GraphBuilder()
        lhs = builder.constant([[1.0, 2.0]], dtype="float32")
        rhs = builder.constant([[3.0], [4.0]], dtype="float32")
        graph = builder.build(builder.matmul(lhs, rhs, name="result"))

        executable = tinyaccel.compile(graph)
        result = executable.run()

        np.testing.assert_array_equal(result, np.array([[11.0]], dtype=np.float32))
        self.assertEqual(executable.last_report.total_cycles, 0)
        self.assertEqual(executable.program.instructions, ())


if __name__ == "__main__":
    unittest.main()
