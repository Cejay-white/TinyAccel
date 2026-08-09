import unittest

import numpy as np

import tinyaccel


class OptimizationPassTests(unittest.TestCase):
    def test_folds_constant_subgraph(self) -> None:
        builder = tinyaccel.GraphBuilder()
        lhs = builder.constant([[1.0, -2.0]], dtype="float32", name="lhs")
        rhs = builder.constant([[3.0], [4.0]], dtype="float32", name="rhs")
        result = builder.relu(builder.matmul(lhs, rhs, name="product"), name="result")
        graph = builder.build(result)

        optimized = tinyaccel.ConstantFoldingPass().run(graph)

        self.assertEqual([op.op for op in optimized.operations], [
            "constant", "constant", "constant", "constant"
        ])
        np.testing.assert_array_equal(
            tinyaccel.evaluate(optimized), np.array([[0.0]], dtype=np.float32)
        )

    def test_simplifies_add_zero_and_eliminates_dead_constant(self) -> None:
        builder = tinyaccel.GraphBuilder()
        value = builder.input("value", (2, 3))
        zero = builder.constant(0.0, dtype="float32", name="zero")
        added = builder.add(value, zero, name="added")
        dead = builder.constant(7.0, dtype="float32", name="dead")
        graph = builder.build(builder.relu(added, name="result"))

        optimized, trace = tinyaccel.default_pipeline().run_with_trace(graph)

        self.assertEqual([result.pass_name for result in trace], [
            "constant-folding",
            "algebraic-simplification",
            "matmul-bias-relu-fusion",
            "dead-code-elimination",
        ])
        self.assertEqual([op.op for op in optimized.operations], ["relu"])
        self.assertIs(optimized.operations[0].inputs[0], value)
        self.assertNotIn(dead, optimized.values)

        data = np.arange(6, dtype=np.float32).reshape(2, 3) - 2
        np.testing.assert_array_equal(
            tinyaccel.evaluate(optimized, data), np.maximum(data, 0)
        )

    def test_dce_preserves_required_dependency_chain(self) -> None:
        builder = tinyaccel.GraphBuilder()
        value = builder.input("value", (2,))
        first = builder.relu(value, name="first")
        output = builder.relu(first, name="output")
        builder.relu(value, name="unused")
        graph = builder.build(output)

        optimized = tinyaccel.DeadCodeEliminationPass().run(graph)

        self.assertEqual([op.output.name for op in optimized.operations], ["first", "output"])

    def test_fuses_single_use_matmul_bias_relu_chain(self) -> None:
        builder = tinyaccel.GraphBuilder()
        lhs = builder.input("lhs", (4, 3))
        rhs = builder.input("rhs", (3, 5))
        bias = builder.input("bias", (5,))
        product = builder.matmul(lhs, rhs)
        added = builder.add(product, bias)
        graph = builder.build(builder.relu(added, name="result"))

        fused = tinyaccel.MatmulBiasReluFusionPass().run(graph)

        self.assertEqual([op.op for op in fused.operations], ["matmul_bias_relu"])
        self.assertEqual(fused.operations[0].inputs, (lhs, rhs, bias))
        self.assertEqual(str(tinyaccel.parse_graph(str(fused))), str(fused))

    def test_does_not_fuse_when_matmul_has_another_user(self) -> None:
        builder = tinyaccel.GraphBuilder()
        lhs = builder.input("lhs", (4, 3))
        rhs = builder.input("rhs", (3, 5))
        bias = builder.input("bias", (5,))
        product = builder.matmul(lhs, rhs)
        added = builder.add(product, bias)
        output = builder.relu(added)
        graph = builder.build((output, product))

        fused = tinyaccel.MatmulBiasReluFusionPass().run(graph)

        self.assertNotIn("matmul_bias_relu", [op.op for op in fused.operations])

    def test_does_not_fuse_broadcast_that_expands_matmul_output(self) -> None:
        builder = tinyaccel.GraphBuilder()
        lhs = builder.input("lhs", (1, 3))
        rhs = builder.input("rhs", (3, 5))
        bias = builder.input("bias", (4, 1))
        product = builder.matmul(lhs, rhs)
        added = builder.add(product, bias)
        graph = builder.build(builder.relu(added))

        fused = tinyaccel.MatmulBiasReluFusionPass().run(graph)

        self.assertNotIn("matmul_bias_relu", [op.op for op in fused.operations])


if __name__ == "__main__":
    unittest.main()
