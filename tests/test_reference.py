import unittest

import numpy as np

import tinyaccel


class ReferenceExecutorTests(unittest.TestCase):
    def test_executes_matmul_bias_relu(self) -> None:
        builder = tinyaccel.GraphBuilder()
        lhs = builder.input("lhs", (5, 3))
        rhs = builder.input("rhs", (3, 4))
        bias = builder.constant(
            np.array([-2.0, -1.0, 1.0, 2.0], dtype=np.float32), name="bias"
        )
        product = builder.matmul(lhs, rhs)
        result = builder.relu(builder.add(product, bias))
        graph = builder.build(result)

        rng = np.random.default_rng(4)
        lhs_data = rng.standard_normal((5, 3), dtype=np.float32)
        rhs_data = rng.standard_normal((3, 4), dtype=np.float32)

        actual = tinyaccel.evaluate(graph, lhs_data, rhs_data)
        expected = np.maximum(
            lhs_data @ rhs_data
            + np.array([-2.0, -1.0, 1.0, 2.0], dtype=np.float32),
            0,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)

    def test_returns_multiple_outputs(self) -> None:
        builder = tinyaccel.GraphBuilder()
        value = builder.input("value", (2,))
        positive = builder.relu(value)
        graph = builder.build((value, positive))
        data = np.array([-1.0, 2.0], dtype=np.float32)

        original, result = tinyaccel.evaluate(graph, value=data)

        np.testing.assert_array_equal(original, data)
        np.testing.assert_array_equal(result, np.array([0.0, 2.0], dtype=np.float32))

    def test_validates_inputs(self) -> None:
        builder = tinyaccel.GraphBuilder()
        value = builder.input("value", (2,))
        graph = builder.build(value)

        with self.assertRaisesRegex(ValueError, "dtype"):
            tinyaccel.evaluate(graph, np.ones((2,), dtype=np.float64))


if __name__ == "__main__":
    unittest.main()
