"""Composable optimization passes for TinyAccel graph IR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .ir import Graph, Operation, TensorType, Value, layout_permutation


class GraphPass(Protocol):
    """Interface implemented by graph-to-graph transformations."""

    @property
    def name(self) -> str: ...

    def run(self, graph: Graph) -> Graph: ...


@dataclass(frozen=True)
class PassResult:
    pass_name: str
    graph: Graph


class PassManager:
    def __init__(self, passes: tuple[GraphPass, ...] | list[GraphPass]) -> None:
        self.passes = tuple(passes)

    def run(self, graph: Graph) -> Graph:
        current = graph
        for graph_pass in self.passes:
            current = graph_pass.run(current)
            current.validate()
        return current

    def run_with_trace(self, graph: Graph) -> tuple[Graph, tuple[PassResult, ...]]:
        current = graph
        results: list[PassResult] = []
        for graph_pass in self.passes:
            current = graph_pass.run(current)
            current.validate()
            results.append(PassResult(graph_pass.name, current))
        return current, tuple(results)


class CanonicalizeConv2dLayoutsPass:
    """Lower NCHW/OIHW Conv2D into the canonical NHWC/HWIO graph form."""

    name = "canonicalize-conv2d-layouts"

    def run(self, graph: Graph) -> Graph:
        reserved_names = {value.name for value in graph.values}

        def fresh_name(stem: str) -> str:
            candidate = stem
            suffix = 0
            while candidate in reserved_names:
                suffix += 1
                candidate = f"{stem}_{suffix}"
            reserved_names.add(candidate)
            return candidate

        operations: list[Operation] = []
        for operation in graph.operations:
            if operation.op != "conv2d":
                operations.append(operation)
                continue

            input_value, weight = operation.inputs
            layouts = (input_value.type.layout, weight.type.layout)
            if layouts == ("NHWC", "HWIO"):
                operations.append(operation)
                continue
            if layouts != ("NCHW", "OIHW"):
                raise ValueError(
                    "conv2d canonicalization requires NHWC/HWIO or NCHW/OIHW"
                )

            input_transform = _make_layout_transform(
                input_value,
                "NHWC",
                fresh_name(f"{operation.output.name}_input_nhwc"),
            )
            weight_transform = _make_layout_transform(
                weight,
                "HWIO",
                fresh_name(f"{operation.output.name}_weight_hwio"),
            )
            canonical_shape = tuple(
                operation.output.type.shape[axis]
                for axis in layout_permutation("NCHW", "NHWC")
            )
            canonical_output = Value(
                fresh_name(f"{operation.output.name}_nhwc"),
                TensorType(canonical_shape, operation.output.type.dtype, "NHWC"),
            )
            canonical_conv2d = Operation(
                "conv2d",
                (input_transform.output, weight_transform.output),
                canonical_output,
                operation.attributes,
            )
            output_transform = Operation(
                "layout_transform",
                (canonical_output,),
                operation.output,
                {"target_layout": "NCHW"},
            )
            operations.extend(
                (
                    input_transform,
                    weight_transform,
                    canonical_conv2d,
                    output_transform,
                )
            )

        return Graph(graph.inputs, operations, graph.outputs)


class ConstantFoldingPass:
    name = "constant-folding"

    def run(self, graph: Graph) -> Graph:
        constants: dict[Value, np.ndarray] = {}
        operations: list[Operation] = []

        for operation in graph.operations:
            if operation.op == "constant":
                value = np.asarray(
                    operation.attributes["value"], dtype=operation.output.type.dtype
                )
                constants[operation.output] = value
                operations.append(operation)
                continue

            if (
                operation.op
                in {
                    "matmul",
                    "add",
                    "relu",
                    "layout_transform",
                    "matmul_bias_relu",
                }
                and operation.inputs
                and all(value in constants for value in operation.inputs)
            ):
                inputs = [constants[value] for value in operation.inputs]
                folded = _evaluate_operation(operation, inputs)
                folded = np.asarray(folded, dtype=operation.output.type.dtype)
                if folded.shape != operation.output.type.shape:
                    raise RuntimeError(
                        f"constant folding {operation.op!r} produced {folded.shape}, "
                        f"expected {operation.output.type.shape}"
                    )
                replacement = Operation(
                    "constant",
                    (),
                    operation.output,
                    {"value": folded.copy()},
                )
                constants[operation.output] = folded
                operations.append(replacement)
            else:
                operations.append(operation)

        return Graph(graph.inputs, operations, graph.outputs)


class AlgebraicSimplificationPass:
    name = "algebraic-simplification"

    def run(self, graph: Graph) -> Graph:
        replacements: dict[Value, Value] = {}
        constants: dict[Value, np.ndarray] = {}
        operations: list[Operation] = []

        def resolve(value: Value) -> Value:
            while value in replacements:
                value = replacements[value]
            return value

        for operation in graph.operations:
            inputs = tuple(resolve(value) for value in operation.inputs)
            if operation.op == "constant":
                constants[operation.output] = np.asarray(
                    operation.attributes["value"], dtype=operation.output.type.dtype
                )

            if operation.op == "add":
                replacement = _identity_add_operand(operation.output, inputs, constants)
                if replacement is not None:
                    replacements[operation.output] = replacement
                    continue

            if inputs != operation.inputs:
                operation = Operation(
                    operation.op, inputs, operation.output, operation.attributes
                )
            operations.append(operation)

        outputs = tuple(resolve(value) for value in graph.outputs)
        return Graph(graph.inputs, operations, outputs)


class LayoutTransformSimplificationPass:
    """Eliminate identity and adjacent inverse layout transformations."""

    name = "layout-transform-simplification"

    def run(self, graph: Graph) -> Graph:
        replacements: dict[Value, Value] = {}
        producers: dict[Value, Operation] = {}
        operations: list[Operation] = []

        def resolve(value: Value) -> Value:
            while value in replacements:
                value = replacements[value]
            return value

        for operation in graph.operations:
            inputs = tuple(resolve(value) for value in operation.inputs)
            if operation.op == "layout_transform":
                source = inputs[0]
                if source.type == operation.output.type:
                    replacements[operation.output] = source
                    continue
                producer = producers.get(source)
                if (
                    producer is not None
                    and producer.op == "layout_transform"
                    and producer.inputs[0].type == operation.output.type
                ):
                    replacements[operation.output] = producer.inputs[0]
                    continue

            if inputs != operation.inputs:
                operation = Operation(
                    operation.op, inputs, operation.output, operation.attributes
                )
            operations.append(operation)
            producers[operation.output] = operation

        outputs = tuple(resolve(value) for value in graph.outputs)
        return Graph(graph.inputs, operations, outputs)


class DeadCodeEliminationPass:
    name = "dead-code-elimination"

    def run(self, graph: Graph) -> Graph:
        live_values = set(graph.outputs)
        kept: list[Operation] = []
        for operation in reversed(graph.operations):
            if operation.output in live_values:
                kept.append(operation)
                live_values.update(operation.inputs)
        kept.reverse()
        return Graph(graph.inputs, kept, graph.outputs)


class MatmulBiasReluFusionPass:
    name = "matmul-bias-relu-fusion"

    def run(self, graph: Graph) -> Graph:
        skip: set[Value] = set()
        replacements: dict[Value, Operation] = {}
        graph_outputs = set(graph.outputs)

        for matmul in graph.operations:
            if matmul.op != "matmul" or matmul.output in graph_outputs:
                continue
            matmul_users = graph.users(matmul.output)
            if len(matmul_users) != 1:
                continue
            add = matmul_users[0]
            if add.op != "add" or add.output in graph_outputs:
                continue
            if add.output.type != matmul.output.type:
                continue
            add_users = graph.users(add.output)
            if len(add_users) != 1 or add_users[0].op != "relu":
                continue
            relu = add_users[0]
            product_uses = sum(value == matmul.output for value in add.inputs)
            if product_uses != 1:
                continue
            if add.inputs[0] == matmul.output:
                bias = add.inputs[1]
            elif add.inputs[1] == matmul.output:
                bias = add.inputs[0]
            else:
                continue

            fused = Operation(
                "matmul_bias_relu",
                (matmul.inputs[0], matmul.inputs[1], bias),
                relu.output,
            )
            skip.update((matmul.output, add.output))
            replacements[relu.output] = fused

        operations: list[Operation] = []
        for operation in graph.operations:
            if operation.output in skip:
                continue
            operations.append(replacements.get(operation.output, operation))
        return Graph(graph.inputs, operations, graph.outputs)


def default_pipeline() -> PassManager:
    """Return the small, deterministic optimization pipeline."""

    return PassManager(
        [
            CanonicalizeConv2dLayoutsPass(),
            ConstantFoldingPass(),
            AlgebraicSimplificationPass(),
            LayoutTransformSimplificationPass(),
            MatmulBiasReluFusionPass(),
            DeadCodeEliminationPass(),
        ]
    )


def _evaluate_operation(
    operation: Operation, inputs: list[np.ndarray]
) -> np.ndarray:
    if operation.op == "matmul":
        return inputs[0] @ inputs[1]
    if operation.op == "add":
        return inputs[0] + inputs[1]
    if operation.op == "relu":
        return np.maximum(inputs[0], 0)
    if operation.op == "layout_transform":
        permutation = layout_permutation(
            operation.inputs[0].type.layout,
            operation.output.type.layout,
        )
        return np.transpose(inputs[0], permutation)
    if operation.op == "matmul_bias_relu":
        return np.maximum(inputs[0] @ inputs[1] + inputs[2], 0)
    raise NotImplementedError(
        f"constant folding does not support {operation.op!r}"
    )


def _identity_add_operand(
    output: Value,
    inputs: tuple[Value, ...],
    constants: dict[Value, np.ndarray],
) -> Value | None:
    if len(inputs) != 2:
        return None
    lhs, rhs = inputs
    if rhs in constants and not np.any(constants[rhs]) and lhs.type == output.type:
        return lhs
    if lhs in constants and not np.any(constants[lhs]) and rhs.type == output.type:
        return rhs
    return None


def _make_layout_transform(
    source: Value, target_layout: str, output_name: str
) -> Operation:
    permutation = layout_permutation(source.type.layout, target_layout)
    output_shape = tuple(source.type.shape[axis] for axis in permutation)
    output = Value(
        output_name,
        TensorType(output_shape, source.type.dtype, target_layout),
    )
    return Operation(
        "layout_transform",
        (source,),
        output,
        {"target_layout": target_layout},
    )
