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


if __name__ == "__main__":
    unittest.main()
