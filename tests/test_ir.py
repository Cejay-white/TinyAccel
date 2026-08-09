import unittest

import numpy as np

from tinyaccel import Graph, GraphBuilder, TensorType, parse_graph


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


class GraphParserTests(unittest.TestCase):
    def test_round_trips_canonical_ir(self) -> None:
        builder = GraphBuilder()
        lhs = builder.input("lhs", (4, 8))
        rhs = builder.input("rhs", (8, 3))
        original = builder.build(builder.matmul(lhs, rhs, name="result"))

        parsed = parse_graph(str(original))

        self.assertEqual(str(parsed), str(original))
        self.assertEqual(str(Graph.parse(str(original))), str(original))

    def test_rejects_incorrect_declared_type(self) -> None:
        text = """graph (%a: tensor<2x3xfloat32>, %b: tensor<3x4xfloat32>) {
          %result = matmul(%a, %b) : tensor<2x5xfloat32>
          return %result
        }"""

        with self.assertRaisesRegex(ValueError, "does not match inferred type"):
            parse_graph(text)

    def test_rejects_undefined_operand(self) -> None:
        text = """graph (%a: tensor<2x3xfloat32>) {
          %result = matmul(%a, %missing) : tensor<2x4xfloat32>
          return %result
        }"""

        with self.assertRaisesRegex(ValueError, "undefined operand"):
            parse_graph(text)

    def test_requires_percent_prefixed_value_references(self) -> None:
        text = """graph (%a: tensor<2x3xfloat32>, %b: tensor<3x4xfloat32>) {
          %result = matmul(a, %b) : tensor<2x4xfloat32>
          return %result
        }"""

        with self.assertRaisesRegex(ValueError, "invalid value reference"):
            parse_graph(text)


if __name__ == "__main__":
    unittest.main()
