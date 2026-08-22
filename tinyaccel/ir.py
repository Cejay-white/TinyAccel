"""A deliberately small, SSA-like graph intermediate representation."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from typing import Any, Iterable, Mapping

import numpy as np


_IDENTIFIER = r"[A-Za-z_][A-Za-z0-9_]*"
_TENSOR_TYPE_SYNTAX = r"tensor<[^>]+>"
_LAYOUTS = frozenset({"NCHW", "NHWC", "OIHW", "HWIO"})
_LAYOUT_PERMUTATIONS = {
    ("NCHW", "NHWC"): (0, 2, 3, 1),
    ("NHWC", "NCHW"): (0, 3, 1, 2),
    ("OIHW", "HWIO"): (2, 3, 1, 0),
    ("HWIO", "OIHW"): (3, 2, 0, 1),
}


def layout_permutation(
    source_layout: str | None, target_layout: str
) -> tuple[int, int, int, int]:
    """Return the physical-axis permutation for a supported layout change."""

    if source_layout is None:
        raise ValueError("layout_transform requires a source layout")
    source = str(source_layout).upper()
    target = str(target_layout).upper()
    try:
        return _LAYOUT_PERMUTATIONS[(source, target)]
    except KeyError as error:
        raise ValueError(
            f"unsupported layout transform: {source} -> {target}"
        ) from error


@dataclass(frozen=True)
class TensorType:
    """Static tensor metadata carried by every IR value."""

    shape: tuple[int, ...]
    dtype: np.dtype = field(default_factory=lambda: np.dtype("float32"))
    layout: str | None = None

    def __post_init__(self) -> None:
        shape = tuple(self.shape)
        if any(not isinstance(dim, int) or dim <= 0 for dim in shape):
            raise ValueError(f"tensor dimensions must be positive integers, got {shape}")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "dtype", np.dtype(self.dtype))
        if self.layout is not None:
            layout = str(self.layout).upper()
            if layout not in _LAYOUTS:
                raise ValueError(f"unsupported tensor layout: {self.layout!r}")
            if len(shape) != 4:
                raise ValueError(f"layout {layout} requires a rank-4 tensor")
            object.__setattr__(self, "layout", layout)

    def __str__(self) -> str:
        prefix = "x".join(str(dim) for dim in self.shape)
        if prefix:
            prefix += "x"
        layout = "" if self.layout is None else f", layout={self.layout}"
        return f"tensor<{prefix}{self.dtype.name}{layout}>"


@dataclass(frozen=True)
class Value:
    """An SSA value produced by an input or operation."""

    name: str
    type: TensorType

    def __str__(self) -> str:
        return f"%{self.name}"


@dataclass(frozen=True)
class Operation:
    """A single operation in a graph."""

    op: str
    inputs: tuple[Value, ...]
    output: Value
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "inputs", tuple(self.inputs))
        object.__setattr__(self, "attributes", dict(self.attributes))


class Graph:
    """An ordered, single-block graph with SSA values."""

    def __init__(
        self,
        inputs: Iterable[Value],
        operations: Iterable[Operation],
        outputs: Iterable[Value],
    ) -> None:
        self.inputs = tuple(inputs)
        self.operations = tuple(operations)
        self.outputs = tuple(outputs)
        self._producers: dict[Value, Operation] = {}
        self._users: dict[Value, list[Operation]] = {}
        self.validate()
        for operation in self.operations:
            self._producers[operation.output] = operation
            for value in set(operation.inputs):
                self._users.setdefault(value, []).append(operation)

    def validate(self) -> None:
        if not self.outputs:
            raise ValueError("graph must have at least one output")

        defined: dict[str, Value] = {}
        for value in self.inputs:
            if value.name in defined:
                raise ValueError(f"duplicate value name: %{value.name}")
            defined[value.name] = value

        for operation in self.operations:
            if re.fullmatch(_IDENTIFIER, operation.op) is None:
                raise ValueError(f"invalid operation name: {operation.op!r}")
            for value in operation.inputs:
                if defined.get(value.name) != value:
                    raise ValueError(f"use of undefined value: %{value.name}")
            if operation.output.name in defined:
                raise ValueError(f"duplicate value name: %{operation.output.name}")
            defined[operation.output.name] = operation.output

        for value in self.outputs:
            if defined.get(value.name) != value:
                raise ValueError(f"undefined graph output: %{value.name}")

    @property
    def values(self) -> tuple[Value, ...]:
        return self.inputs + tuple(operation.output for operation in self.operations)

    def producer(self, value: Value) -> Operation | None:
        return self._producers.get(value)

    def users(self, value: Value) -> tuple[Operation, ...]:
        return tuple(self._users.get(value, ()))

    def __str__(self) -> str:
        arguments = ", ".join(f"{value}: {value.type}" for value in self.inputs)
        lines = [f"graph ({arguments}) {{"]
        for operation in self.operations:
            operands = ", ".join(str(value) for value in operation.inputs)
            attributes = _format_attributes(operation.attributes)
            lines.append(
                f"  {operation.output} = {operation.op}({operands}){attributes} : "
                f"{operation.output.type}"
            )
        outputs = ", ".join(str(value) for value in self.outputs)
        lines.append(f"  return {outputs}")
        lines.append("}")
        return "\n".join(lines)

    @classmethod
    def parse(cls, text: str) -> Graph:
        """Parse the canonical text produced by :meth:`__str__`."""

        return parse_graph(text)

    def to_dot(self) -> str:
        """Return a dependency graph in GraphViz DOT format."""

        output_values = set(self.outputs)
        lines = ["digraph TinyAccel {", "  rankdir=LR;"]
        for value in self.values:
            shape = "doublecircle" if value in output_values else "ellipse"
            lines.append(
                f'  value_{value.name} [label="%{value.name}\\n{value.type}", '
                f"shape={shape}];"
            )
        for index, operation in enumerate(self.operations):
            lines.append(f'  op_{index} [label="{operation.op}", shape=box];')
            for value in operation.inputs:
                lines.append(f"  value_{value.name} -> op_{index};")
            lines.append(f"  op_{index} -> value_{operation.output.name};")
        lines.append("}")
        return "\n".join(lines)


class GraphBuilder:
    """Convenience builder for constructing valid TinyAccel graphs."""

    def __init__(self) -> None:
        self._inputs: list[Value] = []
        self._operations: list[Operation] = []
        self._names: set[str] = set()
        self._next_value_id = 0

    def input(
        self,
        name: str,
        shape: Iterable[int],
        dtype: str | np.dtype = "float32",
        layout: str | None = None,
    ) -> Value:
        self._reserve_name(name)
        value = Value(name, TensorType(tuple(shape), np.dtype(dtype), layout))
        self._inputs.append(value)
        return value

    def constant(
        self,
        value: Any,
        *,
        dtype: str | np.dtype | None = None,
        name: str | None = None,
        layout: str | None = None,
    ) -> Value:
        array = np.asarray(value, dtype=dtype)
        if array.dtype.kind not in "biuf":
            raise ValueError(f"unsupported constant dtype: {array.dtype}")
        output = self._new_output(name, TensorType(array.shape, array.dtype, layout))
        self._operations.append(
            Operation("constant", (), output, {"value": array.copy()})
        )
        return output

    def matmul(self, lhs: Value, rhs: Value, name: str | None = None) -> Value:
        self._require_defined(lhs, rhs)
        if len(lhs.type.shape) != 2 or len(rhs.type.shape) != 2:
            raise ValueError("matmul currently requires rank-2 tensors")
        if lhs.type.shape[1] != rhs.type.shape[0]:
            raise ValueError(
                "matmul dimension mismatch: "
                f"{lhs.type.shape} cannot be multiplied by {rhs.type.shape}"
            )
        self._require_same_dtype("matmul", lhs, rhs)
        output = self._new_output(
            name,
            TensorType((lhs.type.shape[0], rhs.type.shape[1]), lhs.type.dtype),
        )
        self._operations.append(Operation("matmul", (lhs, rhs), output))
        return output

    def add(self, lhs: Value, rhs: Value, name: str | None = None) -> Value:
        self._require_defined(lhs, rhs)
        self._require_same_dtype("add", lhs, rhs)
        if (
            lhs.type.layout is not None
            and rhs.type.layout is not None
            and lhs.type.layout != rhs.type.layout
        ):
            raise ValueError(
                f"add layout mismatch: {lhs.type.layout!r} and {rhs.type.layout!r}"
            )
        try:
            shape = np.broadcast_shapes(lhs.type.shape, rhs.type.shape)
        except ValueError as error:
            raise ValueError(
                f"add shapes are not broadcastable: {lhs.type.shape} and "
                f"{rhs.type.shape}"
            ) from error
        layout = lhs.type.layout or rhs.type.layout
        output = self._new_output(name, TensorType(shape, lhs.type.dtype, layout))
        self._operations.append(Operation("add", (lhs, rhs), output))
        return output

    def relu(self, value: Value, name: str | None = None) -> Value:
        self._require_defined(value)
        output = self._new_output(name, value.type)
        self._operations.append(Operation("relu", (value,), output))
        return output

    def layout_transform(
        self,
        value: Value,
        target_layout: str,
        name: str | None = None,
    ) -> Value:
        """Reorder a rank-4 tensor between supported activation/weight layouts."""

        self._require_defined(value)
        target = str(target_layout).upper()
        permutation = layout_permutation(value.type.layout, target)
        output_shape = tuple(value.type.shape[axis] for axis in permutation)
        output = self._new_output(
            name,
            TensorType(output_shape, value.type.dtype, target),
        )
        self._operations.append(
            Operation(
                "layout_transform",
                (value,),
                output,
                {"target_layout": target},
            )
        )
        return output

    def conv2d(
        self,
        input_value: Value,
        weight: Value,
        *,
        stride: int | Iterable[int] = (1, 1),
        padding: int | Iterable[int] = (0, 0, 0, 0),
        dilation: int | Iterable[int] = (1, 1),
        name: str | None = None,
    ) -> Value:
        """Build a float32 NHWC x HWIO two-dimensional convolution."""

        self._require_defined(input_value, weight)
        self._require_same_dtype("conv2d", input_value, weight)
        if input_value.type.dtype != np.dtype("float32"):
            raise ValueError("conv2d currently requires float32 tensors")
        if input_value.type.layout != "NHWC" or weight.type.layout != "HWIO":
            raise ValueError("conv2d currently requires NHWC input and HWIO weight")
        stride_pair = _normalize_pair("stride", stride)
        dilation_pair = _normalize_pair("dilation", dilation)
        padding_quad = _normalize_padding(padding)
        n_size, input_h, input_w, input_c = input_value.type.shape
        kernel_h, kernel_w, weight_c, output_c = weight.type.shape
        if input_c != weight_c:
            raise ValueError(
                f"conv2d input channels mismatch: {input_c} and {weight_c}"
            )
        effective_h = (kernel_h - 1) * dilation_pair[0] + 1
        effective_w = (kernel_w - 1) * dilation_pair[1] + 1
        padded_h = input_h + padding_quad[0] + padding_quad[1]
        padded_w = input_w + padding_quad[2] + padding_quad[3]
        if padded_h < effective_h or padded_w < effective_w:
            raise ValueError("conv2d effective kernel exceeds padded input")
        output_h = (padded_h - effective_h) // stride_pair[0] + 1
        output_w = (padded_w - effective_w) // stride_pair[1] + 1
        output = self._new_output(
            name,
            TensorType((n_size, output_h, output_w, output_c), "float32", "NHWC"),
        )
        self._operations.append(
            Operation(
                "conv2d",
                (input_value, weight),
                output,
                {
                    "stride": stride_pair,
                    "padding": padding_quad,
                    "dilation": dilation_pair,
                },
            )
        )
        return output

    def matmul_bias_relu(
        self,
        lhs: Value,
        rhs: Value,
        bias: Value,
        name: str | None = None,
    ) -> Value:
        """Build the canonical fused MatMul + bias + ReLU operation."""

        self._require_defined(lhs, rhs, bias)
        if len(lhs.type.shape) != 2 or len(rhs.type.shape) != 2:
            raise ValueError("matmul_bias_relu requires rank-2 matmul inputs")
        if lhs.type.shape[1] != rhs.type.shape[0]:
            raise ValueError(
                "matmul_bias_relu dimension mismatch: "
                f"{lhs.type.shape} cannot be multiplied by {rhs.type.shape}"
            )
        self._require_same_dtype("matmul_bias_relu", lhs, rhs)
        self._require_same_dtype("matmul_bias_relu", lhs, bias)
        matmul_shape = (lhs.type.shape[0], rhs.type.shape[1])
        try:
            output_shape = np.broadcast_shapes(matmul_shape, bias.type.shape)
        except ValueError as error:
            raise ValueError(
                f"bias shape {bias.type.shape} cannot broadcast to {matmul_shape}"
            ) from error
        if output_shape != matmul_shape:
            raise ValueError(
                f"bias shape {bias.type.shape} expands matmul output {matmul_shape}"
            )
        output = self._new_output(name, TensorType(matmul_shape, lhs.type.dtype))
        self._operations.append(
            Operation("matmul_bias_relu", (lhs, rhs, bias), output)
        )
        return output

    def build(self, outputs: Value | Iterable[Value]) -> Graph:
        graph_outputs = (outputs,) if isinstance(outputs, Value) else tuple(outputs)
        return Graph(self._inputs, self._operations, graph_outputs)

    def _new_output(self, name: str | None, tensor_type: TensorType) -> Value:
        output_name = name or self._fresh_name()
        self._reserve_name(output_name)
        return Value(output_name, tensor_type)

    def _fresh_name(self) -> str:
        while True:
            name = f"v{self._next_value_id}"
            self._next_value_id += 1
            if name not in self._names:
                return name

    def _reserve_name(self, name: str) -> None:
        if re.fullmatch(_IDENTIFIER, name) is None:
            raise ValueError(f"invalid value name: {name!r}")
        if name in self._names:
            raise ValueError(f"duplicate value name: %{name}")
        self._names.add(name)

    def _require_defined(self, *values: Value) -> None:
        known_values = self._inputs + [op.output for op in self._operations]
        for value in values:
            if value not in known_values:
                raise ValueError(f"value {value} does not belong to this builder")

    @staticmethod
    def _require_same_dtype(op: str, lhs: Value, rhs: Value) -> None:
        if lhs.type.dtype != rhs.type.dtype:
            raise ValueError(
                f"{op} dtype mismatch: {lhs.type.dtype} and {rhs.type.dtype}"
            )


def parse_graph(text: str) -> Graph:
    """Parse canonical TinyAccel graph IR and re-run type inference."""

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 3:
        raise ValueError("IR must contain a graph header, return, and closing brace")

    header = re.fullmatch(r"graph \((.*)\) \{", lines[0])
    if header is None or lines[-1] != "}":
        raise ValueError("invalid graph header or missing closing brace")

    builder = GraphBuilder()
    values: dict[str, Value] = {}
    arguments = header.group(1).strip()
    if arguments:
        for argument in _split_graph_arguments(arguments):
            match = re.fullmatch(
                rf"%({_IDENTIFIER})\s*:\s*({_TENSOR_TYPE_SYNTAX})",
                argument.strip(),
            )
            if match is None:
                raise ValueError(f"invalid graph argument: {argument.strip()!r}")
            name, type_text = match.groups()
            tensor_type = _parse_tensor_type(type_text)
            values[name] = builder.input(
                name, tensor_type.shape, tensor_type.dtype, tensor_type.layout
            )

    for line_number, line in enumerate(lines[1:-2], start=2):
        match = re.fullmatch(
            rf"%({_IDENTIFIER})\s*=\s*({_IDENTIFIER})\(([^)]*)\)"
            rf"(?:\s+(\{{.*\}}))?\s*:\s*({_TENSOR_TYPE_SYNTAX})",
            line,
        )
        if match is None:
            raise ValueError(f"invalid operation on line {line_number}: {line!r}")
        output_name, op_name, operand_text, attribute_text, type_text = match.groups()
        operand_names = _parse_value_references(
            operand_text, f"operation on line {line_number}", allow_empty=True
        )
        try:
            operands = [values[name] for name in operand_names]
        except KeyError as error:
            raise ValueError(
                f"undefined operand %{error.args[0]} on line {line_number}"
            ) from error
        attributes = _parse_attributes(attribute_text, line_number)
        declared_type = _parse_tensor_type(type_text)

        if op_name == "constant" and not operands:
            if set(attributes) != {"value"}:
                raise ValueError("constant requires exactly one 'value' attribute")
            result = builder.constant(
                attributes["value"],
                dtype=declared_type.dtype,
                name=output_name,
                layout=declared_type.layout,
            )
        elif op_name == "matmul" and len(operands) == 2 and not attributes:
            result = builder.matmul(operands[0], operands[1], name=output_name)
        elif op_name == "add" and len(operands) == 2 and not attributes:
            result = builder.add(operands[0], operands[1], name=output_name)
        elif op_name == "relu" and len(operands) == 1 and not attributes:
            result = builder.relu(operands[0], name=output_name)
        elif op_name == "layout_transform" and len(operands) == 1:
            if set(attributes) != {"target_layout"}:
                raise ValueError(
                    "layout_transform requires exactly one 'target_layout' attribute"
                )
            result = builder.layout_transform(
                operands[0], attributes["target_layout"], name=output_name
            )
        elif (
            op_name == "matmul_bias_relu"
            and len(operands) == 3
            and not attributes
        ):
            result = builder.matmul_bias_relu(
                operands[0], operands[1], operands[2], name=output_name
            )
        elif op_name == "conv2d" and len(operands) == 2:
            if set(attributes) != {"stride", "padding", "dilation"}:
                raise ValueError(
                    "conv2d requires stride, padding, and dilation attributes"
                )
            result = builder.conv2d(
                operands[0],
                operands[1],
                stride=attributes["stride"],
                padding=attributes["padding"],
                dilation=attributes["dilation"],
                name=output_name,
            )
        else:
            raise ValueError(
                f"unsupported operation {op_name!r} with {len(operands)} operands "
                f"on line {line_number}"
            )

        if result.type != declared_type:
            raise ValueError(
                f"declared type {declared_type} does not match inferred type "
                f"{result.type} on line {line_number}"
            )
        values[output_name] = result

    return_match = re.fullmatch(r"return\s+(.+)", lines[-2])
    if return_match is None:
        raise ValueError("graph must end with a return statement")
    output_names = _parse_value_references(return_match.group(1), "return statement")
    try:
        outputs = [values[name] for name in output_names]
    except KeyError as error:
        raise ValueError(f"undefined graph output %{error.args[0]}") from error
    return builder.build(outputs)


def _parse_tensor_type(text: str) -> TensorType:
    match = re.fullmatch(r"tensor<(.+)>", text)
    if match is None:
        raise ValueError(f"invalid tensor type: {text!r}")
    parts = [part.strip() for part in match.group(1).split(",")]
    if len(parts) > 2 or (len(parts) == 2 and not parts[1].startswith("layout=")):
        raise ValueError(f"invalid tensor type: {text!r}")
    layout = None if len(parts) == 1 else parts[1].split("=", 1)[1]
    type_parts = parts[0].split("x")
    dtype_text = type_parts[-1]
    shape_parts = type_parts[:-1]
    if any(not part.isdigit() for part in shape_parts):
        raise ValueError(f"invalid tensor type: {text!r}")
    shape = tuple(int(dimension) for dimension in shape_parts)
    try:
        dtype = np.dtype(dtype_text)
    except TypeError as error:
        raise ValueError(f"unsupported dtype in tensor type: {text!r}") from error
    return TensorType(shape, dtype, layout)


def _normalize_pair(name: str, value: int | Iterable[int]) -> tuple[int, int]:
    items = (value, value) if isinstance(value, int) else tuple(value)
    if len(items) != 2 or any(not isinstance(item, int) or item <= 0 for item in items):
        raise ValueError(f"{name} must contain two positive integers")
    return int(items[0]), int(items[1])


def _normalize_padding(value: int | Iterable[int]) -> tuple[int, int, int, int]:
    if isinstance(value, int):
        items = (value,) * 4
    else:
        items = tuple(value)
        if len(items) == 2:
            items = (items[0], items[0], items[1], items[1])
    if len(items) != 4 or any(not isinstance(item, int) or item < 0 for item in items):
        raise ValueError("padding must contain two or four non-negative integers")
    return tuple(int(item) for item in items)


def _parse_value_references(
    text: str, context: str, *, allow_empty: bool = False
) -> list[str]:
    if not text.strip() and allow_empty:
        return []
    names: list[str] = []
    for reference in text.split(","):
        match = re.fullmatch(rf"%({_IDENTIFIER})", reference.strip())
        if match is None:
            raise ValueError(f"invalid value reference {reference.strip()!r} in {context}")
        names.append(match.group(1))
    return names


def _split_graph_arguments(text: str) -> list[str]:
    arguments: list[str] = []
    start = 0
    depth = 0
    for index, character in enumerate(text):
        if character == "<":
            depth += 1
        elif character == ">":
            depth -= 1
        elif character == "," and depth == 0:
            arguments.append(text[start:index].strip())
            start = index + 1
    arguments.append(text[start:].strip())
    return arguments


def _format_attributes(attributes: Mapping[str, Any]) -> str:
    if not attributes:
        return ""
    serializable = {key: _to_json_value(value) for key, value in attributes.items()}
    return " " + json.dumps(serializable, sort_keys=True, separators=(",", ":"))


def _parse_attributes(text: str | None, line_number: int) -> dict[str, Any]:
    if text is None:
        return {}
    try:
        attributes = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid attributes on line {line_number}: {error.msg}") from error
    if not isinstance(attributes, dict):
        raise ValueError(f"attributes on line {line_number} must be a JSON object")
    return attributes


def _to_json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {key: _to_json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_to_json_value(item) for item in value]
    return value
