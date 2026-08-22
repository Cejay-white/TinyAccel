import unittest

import numpy as np

import tinyaccel
from tinyaccel.simulator import ExecutionResource


def build_matmul(m: int, k: int, n: int):
    builder = tinyaccel.GraphBuilder()
    lhs = builder.input("lhs", (m, k))
    rhs = builder.input("rhs", (k, n))
    return builder.build(builder.matmul(lhs, rhs, name="result"))


class ResourceTimingTests(unittest.TestCase):
    def test_hardware_requires_boolean_overlap_switch(self) -> None:
        with self.assertRaisesRegex(TypeError, "overlap_resources must be a bool"):
            tinyaccel.HardwareConfig(overlap_resources=1)  # type: ignore[arg-type]

    def test_default_mode_preserves_sequential_timing(self) -> None:
        executable = tinyaccel.compile(
            build_matmul(4, 4, 4),
            options=tinyaccel.CompileOptions(tile_m=4, tile_n=4, tile_k=4),
            hardware=tinyaccel.HardwareConfig(
                dma_bytes_per_cycle=4,
                macs_per_cycle=1,
            ),
        )
        values = np.ones((4, 4), dtype=np.float32)

        executable.run(values, values)
        report = executable.last_report

        self.assertFalse(report.overlap_resources)
        self.assertEqual(report.total_cycles, 128)
        self.assertEqual(report.total_cycles, report.sequential_cycles)
        self.assertEqual(report.overlap_cycles_saved, 0)
        for previous, current in zip(report.timeline, report.timeline[1:]):
            self.assertEqual(previous.end_cycle, current.start_cycle)

    def test_independent_dma_and_compute_resources_overlap(self) -> None:
        executable = tinyaccel.compile(
            build_matmul(4, 4, 4),
            options=tinyaccel.CompileOptions(tile_m=4, tile_n=4, tile_k=4),
            hardware=tinyaccel.HardwareConfig(
                dma_bytes_per_cycle=4,
                macs_per_cycle=1,
                overlap_resources=True,
            ),
        )
        values = np.ones((4, 4), dtype=np.float32)

        actual = executable.run(values, values)
        report = executable.last_report

        np.testing.assert_array_equal(actual, values @ values)
        self.assertTrue(report.overlap_resources)
        self.assertEqual(report.sequential_cycles, 128)
        self.assertEqual(report.total_cycles, 112)
        self.assertEqual(report.overlap_cycles_saved, 16)
        self.assertEqual(
            report.resource_cycles,
            {"DMA": 48, "COMPUTE": 80, "LAYOUT": 0},
        )
        self.assertAlmostEqual(report.resource_utilization["DMA"], 48 / 112)
        self.assertAlmostEqual(report.resource_utilization["COMPUTE"], 80 / 112)

        zero, lhs_load, rhs_load, matmul, store = report.timeline
        self.assertEqual(zero.resource, ExecutionResource.COMPUTE.value)
        self.assertEqual(lhs_load.resource, ExecutionResource.DMA.value)
        self.assertEqual((zero.start_cycle, lhs_load.start_cycle), (0, 0))
        self.assertEqual(lhs_load.end_cycle, rhs_load.start_cycle)
        self.assertEqual(matmul.start_cycle, rhs_load.end_cycle)
        self.assertEqual(store.start_cycle, matmul.end_cycle)
        self.assertIn("resource overlap", executable.report())
        self.assertIn("COMPUTE", executable.timeline(width=32))

    def test_buffer_hazards_prevent_overwriting_live_tiles(self) -> None:
        executable = tinyaccel.compile(
            build_matmul(4, 8, 4),
            options=tinyaccel.CompileOptions(tile_m=4, tile_n=4, tile_k=4),
            hardware=tinyaccel.HardwareConfig(
                dma_bytes_per_cycle=4,
                macs_per_cycle=1,
                overlap_resources=True,
            ),
        )
        values = np.ones((4, 8), dtype=np.float32)
        weights = np.ones((8, 4), dtype=np.float32)

        executable.run(values, weights)
        events = executable.last_report.timeline
        matmuls = [event for event in events if event.opcode == "MATMUL"]
        loads = [event for event in events if event.opcode == "DMA_LOAD"]

        self.assertEqual(len(matmuls), 2)
        self.assertEqual(len(loads), 4)
        self.assertGreaterEqual(loads[2].start_cycle, matmuls[0].end_cycle)
        self.assertGreaterEqual(loads[3].start_cycle, loads[2].end_cycle)
        self.assertGreaterEqual(matmuls[1].start_cycle, loads[3].end_cycle)

    def test_im2col_can_overlap_layout_and_dma_work(self) -> None:
        builder = tinyaccel.GraphBuilder()
        image = builder.input("image", (1, 4, 4, 2), layout="NHWC")
        weight = builder.input("weight", (2, 2, 2, 3), layout="HWIO")
        graph = builder.build(builder.conv2d(image, weight, name="result"))
        executable = tinyaccel.compile(
            graph,
            options=tinyaccel.CompileOptions(
                tile_h=3,
                tile_w=3,
                tile_ic=2,
                tile_oc=3,
                conv2d_lowering="im2col",
            ),
            hardware=tinyaccel.HardwareConfig(
                dma_bytes_per_cycle=4,
                macs_per_cycle=4,
                vector_elements_per_cycle=1,
                overlap_resources=True,
            ),
        )
        rng = np.random.default_rng(31)
        image_data = rng.standard_normal(image.type.shape, dtype=np.float32)
        weight_data = rng.standard_normal(weight.type.shape, dtype=np.float32)

        actual = executable.run(image_data, weight_data)
        expected = tinyaccel.evaluate(graph, image_data, weight_data)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)

        events = executable.last_report.timeline
        im2col = next(event for event in events if event.opcode == "IM2COL")
        weight_load = next(
            event
            for event in events[im2col.instruction_index + 1 :]
            if event.opcode == "DMA_LOAD"
        )
        self.assertEqual(im2col.resource, ExecutionResource.LAYOUT.value)
        self.assertLess(im2col.start_cycle, weight_load.end_cycle)
        self.assertLess(weight_load.start_cycle, im2col.end_cycle)
        self.assertGreater(executable.last_report.overlap_cycles_saved, 0)


if __name__ == "__main__":
    unittest.main()
