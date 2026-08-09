"""NumPy reference execution for validating graph semantics."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from .ir import Graph


class ReferenceExecutor:
    """Execute graph operations directly with NumPy."""

    def run(
        self, graph: Graph, feeds: Mapping[str, np.ndarray]
    ) -> np.ndarray | tuple[np.ndarray, ...]:
        values = _validate_feeds(graph, feeds)
        for operation in graph.operations:
            inputs = [values[value.name] for value in operation.inputs]
            if operation.op == "constant":
                result = np.asarray(
                    operation.attributes["value"], dtype=operation.output.type.dtype
                ).copy()
            elif operation.op == "matmul":
                result = inputs[0] @ inputs[1]
            elif operation.op == "add":
                result = inputs[0] + inputs[1]
            elif operation.op == "relu":
                result = np.maximum(inputs[0], 0)
            elif operation.op == "matmul_bias_relu":
                result = np.maximum(inputs[0] @ inputs[1] + inputs[2], 0)
            else:
                raise NotImplementedError(
                    f"reference executor does not support {operation.op!r}"
                )
            result = np.asarray(result, dtype=operation.output.type.dtype)
            if result.shape != operation.output.type.shape:
                raise RuntimeError(
                    f"operation {operation.op!r} produced shape {result.shape}, "
                    f"expected {operation.output.type.shape}"
                )
            values[operation.output.name] = result

        outputs = tuple(values[value.name] for value in graph.outputs)
        return outputs[0] if len(outputs) == 1 else outputs


def evaluate(
    graph: Graph, *inputs: np.ndarray, **named_inputs: np.ndarray
) -> np.ndarray | tuple[np.ndarray, ...]:
    """Convenience wrapper around :class:`ReferenceExecutor`."""

    if inputs and named_inputs:
        raise TypeError("use positional or named inputs, not both")
    if inputs:
        if len(inputs) != len(graph.inputs):
            raise TypeError(f"expected {len(graph.inputs)} inputs, got {len(inputs)}")
        feeds = {
            value.name: array
            for value, array in zip(graph.inputs, inputs, strict=True)
        }
    else:
        feeds = named_inputs
    return ReferenceExecutor().run(graph, feeds)


def _validate_feeds(
    graph: Graph, feeds: Mapping[str, np.ndarray]
) -> dict[str, np.ndarray]:
    expected_names = {value.name for value in graph.inputs}
    actual_names = set(feeds)
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        extra = sorted(actual_names - expected_names)
        raise ValueError(f"input name mismatch; missing={missing}, extra={extra}")

    validated: dict[str, np.ndarray] = {}
    for value in graph.inputs:
        array = np.asarray(feeds[value.name])
        if array.shape != value.type.shape:
            raise ValueError(
                f"input {value.name!r} has shape {array.shape}, "
                f"expected {value.type.shape}"
            )
        if array.dtype != value.type.dtype:
            raise ValueError(
                f"input {value.name!r} has dtype {array.dtype}, "
                f"expected {value.type.dtype}"
            )
        validated[value.name] = array
    return validated
