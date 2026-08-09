"""Composable optimization passes for TinyAccel graph IR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .ir import Graph, Operation, Value


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

            if operation.inputs and all(value in constants for value in operation.inputs):
                inputs = [constants[value] for value in operation.inputs]
                folded = _evaluate_operation(operation.op, inputs)
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
    """Return the small, deterministic v0.2 optimization pipeline."""

    return PassManager(
        [
            ConstantFoldingPass(),
            AlgebraicSimplificationPass(),
            MatmulBiasReluFusionPass(),
            DeadCodeEliminationPass(),
        ]
    )


def _evaluate_operation(op: str, inputs: list[np.ndarray]) -> np.ndarray:
    if op == "matmul":
        return inputs[0] @ inputs[1]
    if op == "add":
        return inputs[0] + inputs[1]
    if op == "relu":
        return np.maximum(inputs[0], 0)
    if op == "matmul_bias_relu":
        return np.maximum(inputs[0] @ inputs[1] + inputs[2], 0)
    raise NotImplementedError(f"constant folding does not support {op!r}")


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
