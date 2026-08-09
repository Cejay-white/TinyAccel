import unittest

import numpy as np

from tinyaccel import GraphBuilder, TensorType


class TensorTypeTests(unittest.TestCase):
    def test_normalizes_shape_and_dtype(self) -> None:
        tensor_type = TensorType((2, 3), "float32")
        self.assertEqual(tensor_type.shape, (2, 3))
        self.assertEqual(tensor_type.dtype, np.dtype("float32"))
        self.assertEqual(str(tensor_type), "tensor<2x3xfloat32>")

    def test_rejects_invalid_dimensions(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive integers"):
            TensorType((2, 0))


class GraphBuilderTests(unittest.TestCase):
    def test_builds_and_prints_matmul(self) -> None:
        builder = GraphBuilder()
        lhs = builder.input("lhs", (4, 8))
        rhs = builder.input("rhs", (8, 3))
        result = builder.matmul(lhs, rhs, name="result")
        graph = builder.build(result)

        self.assertEqual(result.type.shape, (4, 3))
        self.assertIn("%result = matmul(%lhs, %rhs)", str(graph))

    def test_rejects_incompatible_matmul_shapes(self) -> None:
        builder = GraphBuilder()
        lhs = builder.input("lhs", (4, 8))
        rhs = builder.input("rhs", (7, 3))

        with self.assertRaisesRegex(ValueError, "dimension mismatch"):
            builder.matmul(lhs, rhs)


if __name__ == "__main__":
    unittest.main()

