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
            elif operation.op == "conv2d":
                result = conv2d_nhwc(
                    inputs[0],
                    inputs[1],
                    stride=operation.attributes["stride"],
                    padding=operation.attributes["padding"],
                    dilation=operation.attributes["dilation"],
                )
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


def conv2d_nhwc(
    input_value: np.ndarray,
    weight: np.ndarray,
    *,
    stride: tuple[int, int] = (1, 1),
    padding: tuple[int, int, int, int] = (0, 0, 0, 0),
    dilation: tuple[int, int] = (1, 1),
) -> np.ndarray:
    """Evaluate an NHWC x HWIO convolution without relying on a backend."""

    input_value = np.asarray(input_value)
    weight = np.asarray(weight)
    if input_value.ndim != 4 or weight.ndim != 4:
        raise ValueError("conv2d_nhwc requires rank-4 input and weight")
    n_size, input_h, input_w, input_c = input_value.shape
    kernel_h, kernel_w, weight_c, output_c = weight.shape
    if input_c != weight_c:
        raise ValueError("conv2d_nhwc input channel mismatch")
    stride_h, stride_w = stride
    dilation_h, dilation_w = dilation
    pad_top, pad_bottom, pad_left, pad_right = padding
    effective_h = (kernel_h - 1) * dilation_h + 1
    effective_w = (kernel_w - 1) * dilation_w + 1
    output_h = (input_h + pad_top + pad_bottom - effective_h) // stride_h + 1
    output_w = (input_w + pad_left + pad_right - effective_w) // stride_w + 1
    padded = np.pad(
        input_value,
        ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
    )
    output = np.empty((n_size, output_h, output_w, output_c), dtype=input_value.dtype)
    for n_index in range(n_size):
        for output_y in range(output_h):
            input_y = output_y * stride_h
            for output_x in range(output_w):
                input_x = output_x * stride_w
                patch = padded[
                    n_index,
                    input_y : input_y + effective_h : dilation_h,
                    input_x : input_x + effective_w : dilation_w,
                    :,
                ]
                output[n_index, output_y, output_x, :] = np.tensordot(
                    patch, weight, axes=((0, 1, 2), (0, 1, 2))
                )
    return output
