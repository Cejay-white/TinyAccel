"""A deliberately small, SSA-like graph intermediate representation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class TensorType:
    """Static tensor metadata carried by every IR value."""

    shape: tuple[int, ...]
    dtype: np.dtype = field(default_factory=lambda: np.dtype("float32"))

    def __post_init__(self) -> None:
        shape = tuple(self.shape)
        if not shape or any(not isinstance(dim, int) or dim <= 0 for dim in shape):
            raise ValueError(f"tensor dimensions must be positive integers, got {shape}")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "dtype", np.dtype(self.dtype))

    def __str__(self) -> str:
        dims = "x".join(str(dim) for dim in self.shape)
        return f"tensor<{dims}x{self.dtype.name}>"


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
        self.validate()

    def validate(self) -> None:
        if not self.inputs:
            raise ValueError("graph must have at least one input")
        if not self.outputs:
            raise ValueError("graph must have at least one output")

        defined: dict[str, Value] = {}
        for value in self.inputs:
            if value.name in defined:
                raise ValueError(f"duplicate value name: %{value.name}")
            defined[value.name] = value

        for operation in self.operations:
            for value in operation.inputs:
                if defined.get(value.name) != value:
                    raise ValueError(f"use of undefined value: %{value.name}")
            if operation.output.name in defined:
                raise ValueError(f"duplicate value name: %{operation.output.name}")
            defined[operation.output.name] = operation.output

        for value in self.outputs:
            if defined.get(value.name) != value:
                raise ValueError(f"undefined graph output: %{value.name}")

    def __str__(self) -> str:
        arguments = ", ".join(f"{value}: {value.type}" for value in self.inputs)
        lines = [f"graph ({arguments}) {{"]
        for operation in self.operations:
            operands = ", ".join(str(value) for value in operation.inputs)
            lines.append(
                f"  {operation.output} = {operation.op}({operands}) : "
                f"{operation.output.type}"
            )
        outputs = ", ".join(str(value) for value in self.outputs)
        lines.append(f"  return {outputs}")
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
    ) -> Value:
        self._reserve_name(name)
        value = Value(name, TensorType(tuple(shape), np.dtype(dtype)))
        self._inputs.append(value)
        return value

    def matmul(self, lhs: Value, rhs: Value, name: str | None = None) -> Value:
        self._require_defined(lhs)
        self._require_defined(rhs)
        if len(lhs.type.shape) != 2 or len(rhs.type.shape) != 2:
            raise ValueError("matmul currently requires rank-2 tensors")
        if lhs.type.shape[1] != rhs.type.shape[0]:
            raise ValueError(
                "matmul dimension mismatch: "
                f"{lhs.type.shape} cannot be multiplied by {rhs.type.shape}"
            )
        if lhs.type.dtype != rhs.type.dtype:
            raise ValueError(
                f"matmul dtype mismatch: {lhs.type.dtype} and {rhs.type.dtype}"
            )

        output_name = name or self._fresh_name()
        self._reserve_name(output_name)
        output = Value(
            output_name,
            TensorType((lhs.type.shape[0], rhs.type.shape[1]), lhs.type.dtype),
        )
        self._operations.append(Operation("matmul", (lhs, rhs), output))
        return output

    def build(self, outputs: Value | Iterable[Value]) -> Graph:
        graph_outputs = (outputs,) if isinstance(outputs, Value) else tuple(outputs)
        return Graph(self._inputs, self._operations, graph_outputs)

    def _fresh_name(self) -> str:
        while True:
            name = f"v{self._next_value_id}"
            self._next_value_id += 1
            if name not in self._names:
                return name

    def _reserve_name(self, name: str) -> None:
        if not name or not name.isidentifier():
            raise ValueError(f"invalid value name: {name!r}")
        if name in self._names:
            raise ValueError(f"duplicate value name: %{name}")
        self._names.add(name)

    def _require_defined(self, value: Value) -> None:
        known_values = self._inputs + [op.output for op in self._operations]
        if value not in known_values:
            raise ValueError(f"value {value} does not belong to this builder")

