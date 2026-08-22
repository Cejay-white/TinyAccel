import operator
import sys
import unittest
from dataclasses import dataclass, field
from types import SimpleNamespace
from unittest import mock

import numpy as np

import tinyaccel


def torch_relu(value):
    return value


torch_relu.__module__ = "torch"
torch_relu.__name__ = "relu"


@dataclass(eq=False)
class FakeFxNode:
    op: str
    target: object
    name: str
    args: tuple[object, ...] = ()
    kwargs: dict[str, object] = field(default_factory=dict)


class FakeGraphModule:
    def __init__(self, nodes, **attributes):
        self.graph = SimpleNamespace(nodes=tuple(nodes))
        for name, value in attributes.items():
            setattr(self, name, value)


def build_function_graph():
    lhs = FakeFxNode("placeholder", "lhs", "lhs")
    rhs = FakeFxNode("placeholder", "rhs", "rhs")
    bias = FakeFxNode("get_attr", "parameters.bias", "parameters_bias")
    product = FakeFxNode(
        "call_function", operator.matmul, "product", (lhs, rhs)
    )
    biased = FakeFxNode(
        "call_function", operator.add, "biased", (product, bias)
    )
    result = FakeFxNode(
        "call_function", torch_relu, "result", (biased,), {"inplace": False}
    )
    output = FakeFxNode("output", "output", "output", (result,))
    module = FakeGraphModule(
        (lhs, rhs, bias, product, biased, result, output),
        parameters=SimpleNamespace(
            bias=np.linspace(-1, 1, 4, dtype=np.float32)
        ),
    )
    return module


class TorchFxFrontendTests(unittest.TestCase):
    def test_imports_function_graph_and_compiles_end_to_end(self) -> None:
        graph = tinyaccel.from_torch_fx(
            build_function_graph(),
            {"lhs": (3, 5), "rhs": (5, 4)},
        )

        self.assertEqual(
            [operation.op for operation in graph.operations],
            ["constant", "matmul", "add", "relu"],
        )
        self.assertEqual(graph.operations[0].output.name, "parameters_bias")
        executable = tinyaccel.compile(graph)
        rng = np.random.default_rng(71)
        lhs = rng.standard_normal((3, 5), dtype=np.float32)
        rhs = rng.standard_normal((5, 4), dtype=np.float32)

        actual = executable.run(lhs, rhs)
        expected = np.maximum(
            lhs @ rhs + np.linspace(-1, 1, 4, dtype=np.float32), 0
        )

        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
        self.assertEqual(executable.graph.operations[0].op, "constant")
        self.assertEqual(executable.graph.operations[1].op, "matmul_bias_relu")

    def test_imports_method_calls_and_scalar_literals(self) -> None:
        lhs = FakeFxNode("placeholder", "lhs", "lhs")
        rhs = FakeFxNode("placeholder", "rhs", "rhs")
        product = FakeFxNode("call_method", "matmul", "product", (lhs, rhs))
        shifted = FakeFxNode("call_method", "add", "shifted", (product, 2.0))
        result = FakeFxNode("call_method", "relu", "result", (shifted,))
        output = FakeFxNode("output", "output", "output", (result,))
        graph = tinyaccel.from_torch_fx(
            FakeGraphModule((lhs, rhs, product, shifted, result, output)),
            {
                "lhs": tinyaccel.TensorType((2, 3), "float32"),
                "rhs": np.ones((3, 2), dtype=np.float32),
            },
        )

        self.assertEqual(
            [operation.op for operation in graph.operations],
            ["matmul", "constant", "add", "relu"],
        )
        rng = np.random.default_rng(73)
        lhs_data = rng.standard_normal((2, 3), dtype=np.float32)
        rhs_data = rng.standard_normal((3, 2), dtype=np.float32)
        actual = tinyaccel.evaluate(graph, lhs_data, rhs_data)
        np.testing.assert_allclose(
            actual,
            np.maximum(lhs_data @ rhs_data + 2.0, 0),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_preserves_layout_specs_and_parameter_layouts(self) -> None:
        image = FakeFxNode("placeholder", "image", "image")
        weight = FakeFxNode("get_attr", "weight", "weight")
        output = FakeFxNode("output", "output", "output", ((image, weight),))
        graph = tinyaccel.from_torch_fx(
            FakeGraphModule(
                (image, weight, output),
                weight=np.ones((3, 3, 2, 4), dtype=np.float32),
            ),
            {"image": tinyaccel.TensorType((1, 5, 5, 2), layout="NHWC")},
            parameter_layouts={"weight": "HWIO"},
        )

        self.assertEqual(graph.inputs[0].type.layout, "NHWC")
        self.assertEqual(graph.operations[0].output.type.layout, "HWIO")
        self.assertEqual(graph.outputs, (graph.inputs[0], graph.operations[0].output))

    def test_rejects_missing_specs_and_unsupported_nodes(self) -> None:
        value = FakeFxNode("placeholder", "value", "value")
        output = FakeFxNode("output", "output", "output", (value,))
        with self.assertRaisesRegex(ValueError, "missing input spec"):
            tinyaccel.from_torch_fx(FakeGraphModule((value, output)), {})

        call_module = FakeFxNode("call_module", "relu", "relu", (value,))
        module_output = FakeFxNode(
            "output", "output", "output", (call_module,)
        )
        with self.assertRaisesRegex(NotImplementedError, "call_module"):
            tinyaccel.from_torch_fx(
                FakeGraphModule((value, call_module, module_output)),
                {"value": (2, 3)},
            )

        sine = FakeFxNode("call_function", np.sin, "sine", (value,))
        sine_output = FakeFxNode("output", "output", "output", (sine,))
        with self.assertRaisesRegex(NotImplementedError, "numpy.sin"):
            tinyaccel.from_torch_fx(
                FakeGraphModule((value, sine, sine_output)),
                {"value": (2, 3)},
            )

    def test_rejects_unsupported_add_and_relu_attributes(self) -> None:
        lhs = FakeFxNode("placeholder", "lhs", "lhs")
        rhs = FakeFxNode("placeholder", "rhs", "rhs")
        scaled = FakeFxNode(
            "call_function", operator.add, "scaled", (lhs, rhs), {"alpha": 2}
        )
        scaled_output = FakeFxNode("output", "output", "output", (scaled,))
        with self.assertRaisesRegex(NotImplementedError, "alpha=1"):
            tinyaccel.from_torch_fx(
                FakeGraphModule((lhs, rhs, scaled, scaled_output)),
                {"lhs": (2, 3), "rhs": (2, 3)},
            )

        inplace = FakeFxNode(
            "call_function", torch_relu, "inplace", (lhs,), {"inplace": True}
        )
        inplace_output = FakeFxNode(
            "output", "output", "output", (inplace,)
        )
        with self.assertRaisesRegex(NotImplementedError, "in-place"):
            tinyaccel.from_torch_fx(
                FakeGraphModule((lhs, inplace, inplace_output)),
                {"lhs": (2, 3)},
            )

    def test_trace_entrypoint_reports_missing_optional_dependency(self) -> None:
        with mock.patch.dict(sys.modules, {"torch": None}):
            with self.assertRaisesRegex(
                ModuleNotFoundError, r"install tinyaccel\[torch\]"
            ):
                tinyaccel.trace_torch_module(object(), np.ones((2, 3)))


if __name__ == "__main__":
    unittest.main()
