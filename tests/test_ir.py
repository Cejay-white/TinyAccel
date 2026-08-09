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

    def test_supports_scalar_tensor(self) -> None:
        self.assertEqual(str(TensorType((), "float32")), "tensor<float32>")


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

    def test_builds_multi_operator_graph_and_use_def(self) -> None:
        builder = GraphBuilder()
        lhs = builder.input("lhs", (2, 3))
        rhs = builder.input("rhs", (3, 4))
        bias = builder.constant(np.arange(4, dtype=np.float32), name="bias")
        product = builder.matmul(lhs, rhs, name="product")
        biased = builder.add(product, bias, name="biased")
        result = builder.relu(biased, name="result")
        graph = builder.build(result)

        self.assertEqual(result.type.shape, (2, 4))
        self.assertEqual(graph.producer(product).op, "matmul")
        self.assertEqual(graph.users(product), (graph.operations[2],))
        self.assertIn("digraph TinyAccel", graph.to_dot())
        self.assertIn("product", graph.to_dot())

    def test_add_rejects_non_broadcastable_shapes(self) -> None:
        builder = GraphBuilder()
        lhs = builder.input("lhs", (2, 3))
        rhs = builder.input("rhs", (4,))

        with self.assertRaisesRegex(ValueError, "not broadcastable"):
            builder.add(lhs, rhs)


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

    def test_round_trips_constants_add_and_relu(self) -> None:
        builder = GraphBuilder()
        value = builder.input("value", (2, 3))
        zero = builder.constant(0.0, dtype="float32", name="zero")
        added = builder.add(value, zero, name="added")
        original = builder.build(builder.relu(added, name="result"))

        restored = parse_graph(str(original))

        self.assertEqual(str(restored), str(original))
        self.assertEqual(restored.operations[0].attributes["value"].shape, ())


if __name__ == "__main__":
    unittest.main()
