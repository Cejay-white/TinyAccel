import unittest

import numpy as np

import tinyaccel
from tinyaccel.isa import Opcode
from tinyaccel.simulator import Simulator


class ScheduleTests(unittest.TestCase):
    def test_matmul_schedule_exposes_spatial_and_reduction_loops(self) -> None:
        builder = tinyaccel.GraphBuilder()
        lhs = builder.input("lhs", (17, 11))
        rhs = builder.input("rhs", (11, 9))
        graph = builder.build(builder.matmul(lhs, rhs, name="result"))

        schedule = tinyaccel.create_schedule(
            graph, tile_m=8, tile_n=5, tile_k=4
        )
        operation = schedule.operations[0]

        self.assertEqual(operation.output_tile_shape, (8, 5))
        self.assertEqual(operation.loop("m").tiles, 3)
        self.assertEqual(operation.loop("n").tiles, 2)
        self.assertEqual(operation.loop("k").tiles, 3)
        self.assertEqual(operation.loop("k").kind, "reduction")

    def test_compiler_retains_schedule_used_for_lowering(self) -> None:
        builder = tinyaccel.GraphBuilder()
        lhs = builder.input("lhs", (8, 6))
        rhs = builder.input("rhs", (6, 4))
        graph = builder.build(builder.matmul(lhs, rhs))

        executable = tinyaccel.compile(
            graph,
            options=tinyaccel.CompileOptions(tile_m=3, tile_n=2, tile_k=5),
        )

        scheduled = executable.schedule.operations[0]
        self.assertEqual(
            tuple(loop.tile for loop in scheduled.loops), (3, 2, 5)
        )

    def test_lowering_obeys_schedule_ir_instead_of_compile_options(self) -> None:
        builder = tinyaccel.GraphBuilder()
        lhs = builder.input("lhs", (5, 7))
        rhs = builder.input("rhs", (7, 3))
        graph = builder.build(builder.matmul(lhs, rhs, name="result"))
        scheduled = tinyaccel.Schedule(
            graph,
            (
                tinyaccel.ScheduledOperation(
                    graph.operations[0],
                    (
                        tinyaccel.LoopSpec("m", 5, 2),
                        tinyaccel.LoopSpec("n", 3, 3),
                        tinyaccel.LoopSpec("k", 7, 4, "reduction"),
                    ),
                ),
            ),
        )

        executable = tinyaccel.compile(
            graph,
            options=tinyaccel.CompileOptions(
                tile_m=5, tile_n=1, tile_k=1, optimize=False
            ),
            schedule=scheduled,
        )

        opcodes = [item.opcode for item in executable.program.instructions]
        # The injected schedule has 3 M tiles, 1 N tile, and 2 K tiles.
        self.assertEqual(opcodes.count(Opcode.ZERO), 3)
        self.assertEqual(opcodes.count(Opcode.MATMUL), 6)

    def test_rejects_malformed_matmul_loop_schemas(self) -> None:
        builder = tinyaccel.GraphBuilder()
        lhs = builder.input("lhs", (5, 7))
        rhs = builder.input("rhs", (7, 3))
        graph = builder.build(builder.matmul(lhs, rhs))
        operation = graph.operations[0]
        malformed = (
            (
                "axes",
                (
                    tinyaccel.LoopSpec("n", 3, 3),
                    tinyaccel.LoopSpec("m", 5, 2),
                    tinyaccel.LoopSpec("k", 7, 4, "reduction"),
                ),
            ),
            (
                "extent",
                (
                    tinyaccel.LoopSpec("m", 4, 2),
                    tinyaccel.LoopSpec("n", 3, 3),
                    tinyaccel.LoopSpec("k", 7, 4, "reduction"),
                ),
            ),
            (
                "kind",
                (
                    tinyaccel.LoopSpec("m", 5, 2),
                    tinyaccel.LoopSpec("n", 3, 3),
                    tinyaccel.LoopSpec("k", 7, 4),
                ),
            ),
        )

        for message, loops in malformed:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    tinyaccel.ScheduledOperation(operation, loops)

    def test_rejects_oversized_tiles_and_out_of_order_operations(self) -> None:
        with self.assertRaisesRegex(ValueError, "exceeds extent"):
            tinyaccel.LoopSpec("m", 4, 5)

        builder = tinyaccel.GraphBuilder()
        value = builder.input("value", (4,))
        first = builder.relu(value, name="first")
        graph = builder.build(builder.relu(first, name="output"))
        schedule = tinyaccel.create_schedule(graph)

        with self.assertRaisesRegex(ValueError, "program order"):
            tinyaccel.Schedule(graph, tuple(reversed(schedule.operations)))

    def test_custom_schedule_compile_contract(self) -> None:
        builder = tinyaccel.GraphBuilder()
        lhs = builder.input("lhs", (2, 3))
        rhs = builder.input("rhs", (3, 4))
        graph = builder.build(builder.matmul(lhs, rhs))
        schedule = tinyaccel.create_schedule(graph)

        with self.assertRaisesRegex(ValueError, "optimize=False"):
            tinyaccel.compile(graph, schedule=schedule)

        other_builder = tinyaccel.GraphBuilder()
        other_lhs = other_builder.input("lhs", (2, 3))
        other_rhs = other_builder.input("rhs", (3, 4))
        other_graph = other_builder.build(other_builder.matmul(other_lhs, other_rhs))
        other_schedule = tinyaccel.create_schedule(other_graph)
        with self.assertRaisesRegex(ValueError, "graph being compiled"):
            tinyaccel.compile(
                graph,
                options=tinyaccel.CompileOptions(optimize=False),
                schedule=other_schedule,
            )


class MemoryPlanningTests(unittest.TestCase):
    def test_lifetimes_extend_outputs_and_end_at_last_use(self) -> None:
        builder = tinyaccel.GraphBuilder()
        value = builder.input("value", (4,))
        first = builder.relu(value, name="first")
        second = builder.relu(first, name="second")
        third = builder.add(second, first, name="third")
        output = builder.relu(third, name="output")
        graph = builder.build(output)

        lifetimes = {
            lifetime.value.name: lifetime
            for lifetime in tinyaccel.analyze_lifetimes(graph)
        }

        self.assertEqual((lifetimes["value"].start, lifetimes["value"].end), (-1, 0))
        self.assertEqual((lifetimes["first"].start, lifetimes["first"].end), (0, 2))
        self.assertEqual((lifetimes["output"].start, lifetimes["output"].end), (3, 4))

    def test_plan_reuses_non_overlapping_intermediate_buffers(self) -> None:
        builder = tinyaccel.GraphBuilder()
        value = builder.input("value", (16,))
        first = builder.relu(value, name="first")
        second = builder.relu(first, name="second")
        third = builder.relu(second, name="third")
        output = builder.relu(third, name="output")
        graph = builder.build(output)

        plan = tinyaccel.plan_memory(graph)

        self.assertEqual(
            plan.allocation("first").offset, plan.allocation("third").offset
        )
        self.assertNotEqual(
            plan.allocation("first").offset, plan.allocation("second").offset
        )
        self.assertNotIn("output", plan.allocations)
        materialized = tinyaccel.plan_memory(graph, materialize_outputs=True)
        self.assertIn("output", materialized.allocations)
        self.assertEqual(plan.total_bytes, 128)
        self.assertTrue(
            all(
                allocation.offset % plan.alignment == 0
                for allocation in plan.allocations.values()
            )
        )

    def test_plan_rejects_sram_capacity_overflow(self) -> None:
        builder = tinyaccel.GraphBuilder()
        value = builder.input("value", (17,))
        intermediate = builder.relu(value, name="intermediate")
        graph = builder.build(builder.relu(intermediate, name="output"))

        with self.assertRaisesRegex(ValueError, "requires 128 SRAM bytes"):
            tinyaccel.plan_memory(graph, capacity_bytes=64)

    def test_compile_checks_plan_and_tile_sram_together(self) -> None:
        builder = tinyaccel.GraphBuilder()
        lhs = builder.input("lhs", (2, 2))
        rhs = builder.input("rhs", (2, 2))
        product = builder.matmul(lhs, rhs)
        graph = builder.build(builder.relu(product))

        with self.assertRaisesRegex(ValueError, "48 SRAM bytes plus 64 planned"):
            tinyaccel.compile(
                graph,
                hardware=tinyaccel.HardwareConfig(sram_bytes=111),
            )

    def test_simulator_executes_from_planned_arena(self) -> None:
        builder = tinyaccel.GraphBuilder()
        value = builder.input("value", (3, 5))
        first = builder.relu(value, name="first")
        second = builder.relu(first, name="second")
        third = builder.relu(second, name="third")
        graph = builder.build(builder.relu(third, name="output"))
        executable = tinyaccel.compile(
            graph,
            options=tinyaccel.CompileOptions(optimize=False, tile_m=2, tile_n=3),
        )
        data = np.arange(15, dtype=np.float32).reshape(3, 5) - 7

        np.testing.assert_array_equal(executable.run(data), np.maximum(data, 0))
        self.assertIs(executable.program.memory_plan, executable.memory_plan)
        self.assertEqual(
            executable.memory_plan.allocation("first").offset,
            executable.memory_plan.allocation("third").offset,
        )
        self.assertNotIn("output", executable.memory_plan.allocations)
        self.assertEqual(
            executable.last_report.peak_sram_bytes,
            executable.memory_plan.total_bytes + 2 * 2 * 3 * 4,
        )

    def test_simulator_rejects_peak_above_hardware_capacity(self) -> None:
        builder = tinyaccel.GraphBuilder()
        value = builder.input("value", (3, 5))
        intermediate = builder.relu(value, name="intermediate")
        graph = builder.build(builder.relu(intermediate, name="output"))
        executable = tinyaccel.compile(
            graph,
            options=tinyaccel.CompileOptions(tile_m=2, tile_n=3),
            hardware=tinyaccel.HardwareConfig(sram_bytes=112),
        )
        data = np.ones((3, 5), dtype=np.float32)

        with self.assertRaisesRegex(RuntimeError, "exceeding the 111-byte capacity"):
            Simulator(tinyaccel.HardwareConfig(sram_bytes=111)).run(
                executable.program, {"value": data}
            )


if __name__ == "__main__":
    unittest.main()
