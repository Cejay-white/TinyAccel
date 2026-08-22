"""Optional PyTorch FX importer for TinyAccel's graph IR."""

from __future__ import annotations

import operator
import re
from typing import Any, Mapping

import numpy as np

from .ir import Graph, GraphBuilder, TensorType, Value


def from_torch_fx(
    graph_module: Any,
    input_specs: Mapping[str, Any],
    *,
    parameter_layouts: Mapping[str, str] | None = None,
) -> Graph:
    """Convert a supported ``torch.fx.GraphModule`` into TinyAccel IR.

    This converter intentionally uses only the public FX node protocol and does
    not import PyTorch. ``input_specs`` may contain :class:`TensorType` objects,
    shape tuples, NumPy arrays, or tensor-like values with ``shape`` and
    ``dtype`` attributes.
    """

    try:
        nodes = tuple(graph_module.graph.nodes)
    except AttributeError as error:
        raise TypeError("expected an object with graph.nodes") from error
    if not isinstance(input_specs, Mapping):
        raise TypeError("input_specs must be a mapping keyed by placeholder name")

    layouts = dict(parameter_layouts or {})
    builder = GraphBuilder()
    values: dict[int, Value] = {}
    graph_outputs: tuple[Value, ...] | None = None
    inferred_layouts = _infer_placeholder_layouts(graph_module, nodes)

    for node in nodes:
        node_op = str(getattr(node, "op", ""))
        node_name = _safe_name(getattr(node, "name", "value"))
        target = getattr(node, "target", None)
        args = tuple(getattr(node, "args", ()))
        kwargs = dict(getattr(node, "kwargs", {}))

        if node_op == "placeholder":
            spec_key = str(target)
            spec = input_specs.get(spec_key, input_specs.get(node_name))
            if spec is None:
                raise ValueError(
                    f"missing input spec for FX placeholder {spec_key!r}"
                )
            tensor_type = _normalize_input_spec(spec, spec_key)
            if tensor_type.layout is None and id(node) in inferred_layouts:
                tensor_type = TensorType(
                    tensor_type.shape,
                    tensor_type.dtype,
                    inferred_layouts[id(node)],
                )
            values[id(node)] = builder.input(
                node_name,
                tensor_type.shape,
                tensor_type.dtype,
                tensor_type.layout,
            )
            continue

        if node_op == "get_attr":
            if not isinstance(target, str):
                raise TypeError(f"get_attr target must be a string, got {target!r}")
            constant = _as_numpy(_fetch_attr(graph_module, target), target)
            layout = layouts.get(target, layouts.get(node_name))
            values[id(node)] = builder.constant(
                constant,
                name=node_name,
                layout=layout,
            )
            continue

        if node_op in {"call_function", "call_method"}:
            operation = (
                _classify_function_target(target)
                if node_op == "call_function"
                else _classify_method_target(target)
            )
            values[id(node)] = _lower_call(
                builder,
                values,
                operation,
                args,
                kwargs,
                node_name,
            )
            continue

        if node_op == "call_module":
            if not isinstance(target, str):
                raise TypeError(
                    f"call_module target must be a string, got {target!r}"
                )
            values[id(node)] = _lower_module(
                builder,
                values,
                _fetch_attr(graph_module, target),
                target,
                args,
                kwargs,
                node_name,
            )
            continue

        if node_op == "output":
            if len(args) != 1 or kwargs:
                raise ValueError("FX output must contain exactly one positional value")
            graph_outputs = _resolve_outputs(args[0], values)
            continue

        raise NotImplementedError(f"unsupported FX node op {node_op!r}")

    if graph_outputs is None:
        raise ValueError("FX graph has no output node")
    return builder.build(graph_outputs)


def trace_torch_module(
    module: Any,
    *example_inputs: Any,
    input_layouts: Mapping[str, str] | None = None,
    parameter_layouts: Mapping[str, str] | None = None,
) -> Graph:
    """Symbolically trace a PyTorch module and import it into TinyAccel IR."""

    try:
        import torch
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "trace_torch_module requires PyTorch; install tinyaccel[torch]"
        ) from error

    traced = torch.fx.symbolic_trace(module)
    placeholders = tuple(
        node for node in traced.graph.nodes if node.op == "placeholder"
    )
    if len(placeholders) != len(example_inputs):
        raise ValueError(
            f"expected {len(placeholders)} example inputs, got "
            f"{len(example_inputs)}"
        )

    layouts = dict(input_layouts or {})
    input_specs: dict[str, TensorType] = {}
    for node, example in zip(placeholders, example_inputs, strict=True):
        target = str(node.target)
        array = _as_numpy(example, target)
        layout = layouts.get(target, layouts.get(str(node.name)))
        input_specs[target] = TensorType(array.shape, array.dtype, layout)
    return from_torch_fx(
        traced,
        input_specs,
        parameter_layouts=parameter_layouts,
    )


def _lower_call(
    builder: GraphBuilder,
    values: Mapping[int, Value],
    operation: str,
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
    output_name: str,
) -> Value:
    if operation in {"matmul", "add"}:
        if len(args) != 2:
            raise ValueError(f"FX {operation} requires exactly two operands")
        if operation == "matmul" and kwargs:
            raise NotImplementedError("FX matmul keyword arguments are unsupported")
        if operation == "add":
            unknown = set(kwargs) - {"alpha"}
            if unknown:
                raise NotImplementedError(
                    f"FX add keyword arguments are unsupported: {sorted(unknown)}"
                )
            if kwargs.get("alpha", 1) != 1:
                raise NotImplementedError("FX add currently requires alpha=1")
        lhs, rhs = _resolve_binary_operands(
            builder, values, args, output_name
        )
        if operation == "matmul":
            return builder.matmul(lhs, rhs, name=output_name)
        return builder.add(lhs, rhs, name=output_name)

    if operation == "relu":
        if len(args) != 1:
            raise ValueError("FX relu requires exactly one operand")
        unknown = set(kwargs) - {"inplace"}
        if unknown:
            raise NotImplementedError(
                f"FX relu keyword arguments are unsupported: {sorted(unknown)}"
            )
        if kwargs.get("inplace", False):
            raise NotImplementedError("in-place FX relu is unsupported")
        value = _resolve_value(args[0], values)
        return builder.relu(value, name=output_name)

    raise NotImplementedError(f"unsupported FX operation {operation!r}")


def _lower_module(
    builder: GraphBuilder,
    values: Mapping[int, Value],
    module: Any,
    target: str,
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
    output_name: str,
) -> Value:
    kind = _module_kind(module)
    if kind is None:
        raise NotImplementedError(
            f"unsupported FX call_module target {target!r} "
            f"({_qualified_type_name(module)})"
        )
    if len(args) != 1 or kwargs:
        raise ValueError(
            f"FX {kind} module requires one positional tensor and no keywords"
        )
    input_value = _resolve_value(args[0], values)

    if kind == "Linear":
        weight = _as_numpy(_require_attr(module, "weight", target), f"{target}.weight")
        if weight.ndim != 2:
            raise ValueError(
                f"FX Linear weight must be rank 2, got shape {weight.shape}"
            )
        transposed_weight = builder.constant(
            weight.T.copy(),
            name=f"{output_name}_weight",
        )
        bias = getattr(module, "bias", None)
        product_name = output_name if bias is None else f"{output_name}_matmul"
        result = builder.matmul(input_value, transposed_weight, name=product_name)
        if bias is None:
            return result
        bias_value = _as_numpy(bias, f"{target}.bias")
        if bias_value.ndim != 1 or bias_value.shape[0] != weight.shape[0]:
            raise ValueError(
                "FX Linear bias must be rank 1 with one value per output feature"
            )
        bias_constant = builder.constant(
            bias_value,
            name=f"{output_name}_bias",
        )
        return builder.add(result, bias_constant, name=output_name)

    if kind == "Conv2d":
        groups = getattr(module, "groups", 1)
        if groups != 1:
            raise NotImplementedError("FX Conv2d currently requires groups=1")
        padding_mode = str(getattr(module, "padding_mode", "zeros"))
        if padding_mode != "zeros":
            raise NotImplementedError(
                "FX Conv2d currently requires padding_mode='zeros'"
            )
        padding = getattr(module, "padding", (0, 0))
        if isinstance(padding, str):
            raise NotImplementedError(
                "FX Conv2d string padding modes are unsupported"
            )
        weight = _as_numpy(_require_attr(module, "weight", target), f"{target}.weight")
        if weight.ndim != 4:
            raise ValueError(
                f"FX Conv2d weight must be rank 4, got shape {weight.shape}"
            )
        weight_constant = builder.constant(
            weight,
            name=f"{output_name}_weight",
            layout="OIHW",
        )
        bias = getattr(module, "bias", None)
        convolution_name = output_name if bias is None else f"{output_name}_conv"
        result = builder.conv2d(
            input_value,
            weight_constant,
            stride=getattr(module, "stride", (1, 1)),
            padding=padding,
            dilation=getattr(module, "dilation", (1, 1)),
            name=convolution_name,
        )
        if bias is None:
            return result
        bias_value = _as_numpy(bias, f"{target}.bias")
        if bias_value.ndim != 1 or bias_value.shape[0] != weight.shape[0]:
            raise ValueError(
                "FX Conv2d bias must be rank 1 with one value per output channel"
            )
        bias_constant = builder.constant(
            bias_value.reshape(1, -1, 1, 1),
            name=f"{output_name}_bias",
        )
        return builder.add(result, bias_constant, name=output_name)

    if getattr(module, "inplace", False):
        raise NotImplementedError("in-place FX ReLU module is unsupported")
    return builder.relu(input_value, name=output_name)


def _resolve_binary_operands(
    builder: GraphBuilder,
    values: Mapping[int, Value],
    args: tuple[Any, Any],
    output_name: str,
) -> tuple[Value, Value]:
    lhs = values.get(id(args[0]))
    rhs = values.get(id(args[1]))
    if lhs is None and rhs is None:
        raise TypeError("FX binary operation requires at least one tensor value")
    if lhs is None:
        lhs = builder.constant(
            args[0],
            dtype=rhs.type.dtype,
            name=f"{output_name}_lhs",
        )
    if rhs is None:
        rhs = builder.constant(
            args[1],
            dtype=lhs.type.dtype,
            name=f"{output_name}_rhs",
        )
    return lhs, rhs


def _resolve_value(argument: Any, values: Mapping[int, Value]) -> Value:
    try:
        return values[id(argument)]
    except KeyError as error:
        raise TypeError("FX operand must reference another node") from error


def _resolve_outputs(argument: Any, values: Mapping[int, Value]) -> tuple[Value, ...]:
    if id(argument) in values:
        return (values[id(argument)],)
    if isinstance(argument, (tuple, list)):
        if not argument:
            raise ValueError("FX output sequence must not be empty")
        return tuple(_resolve_value(item, values) for item in argument)
    raise TypeError("FX output must reference a node or a flat node sequence")


def _classify_function_target(target: Any) -> str:
    if target is operator.matmul:
        return "matmul"
    if target is operator.add:
        return "add"

    module = str(getattr(target, "__module__", ""))
    name = str(getattr(target, "__name__", target))
    if module == "torch" or module.startswith("torch."):
        if name in {"matmul", "mm"}:
            return "matmul"
        if name == "add":
            return "add"
        if name == "relu":
            return "relu"
    raise NotImplementedError(
        f"unsupported FX call_function target {_target_label(target)!r}"
    )


def _classify_method_target(target: Any) -> str:
    name = str(target)
    if name in {"matmul", "mm", "__matmul__"}:
        return "matmul"
    if name in {"add", "__add__"}:
        return "add"
    if name == "relu":
        return "relu"
    raise NotImplementedError(f"unsupported FX call_method target {name!r}")


def _module_kind(module: Any) -> str | None:
    name = type(module).__name__
    if name in {"Linear", "Conv2d", "ReLU"}:
        return name
    return None


def _infer_placeholder_layouts(
    graph_module: Any,
    nodes: tuple[Any, ...],
) -> dict[int, str]:
    """Infer PyTorch's logical NCHW layout for Conv2d input placeholders."""

    layouts: dict[int, str] = {}

    def visit_input(node: Any) -> None:
        node_op = str(getattr(node, "op", ""))
        if node_op == "placeholder":
            layouts[id(node)] = "NCHW"
            return
        args = tuple(getattr(node, "args", ()))
        if not args:
            return
        if node_op == "call_module":
            target = getattr(node, "target", None)
            if not isinstance(target, str):
                return
            kind = _module_kind(_fetch_attr(graph_module, target))
            if kind in {"Conv2d", "ReLU"}:
                visit_input(args[0])
            return
        if node_op == "call_function":
            try:
                operation = _classify_function_target(getattr(node, "target", None))
            except NotImplementedError:
                return
            if operation == "relu":
                visit_input(args[0])
            return
        if node_op == "call_method":
            try:
                operation = _classify_method_target(getattr(node, "target", None))
            except NotImplementedError:
                return
            if operation == "relu":
                visit_input(args[0])

    for node in nodes:
        if str(getattr(node, "op", "")) != "call_module":
            continue
        target = getattr(node, "target", None)
        if not isinstance(target, str):
            continue
        if _module_kind(_fetch_attr(graph_module, target)) != "Conv2d":
            continue
        args = tuple(getattr(node, "args", ()))
        if args:
            visit_input(args[0])
    return layouts


def _require_attr(value: Any, name: str, target: str) -> Any:
    if not hasattr(value, name):
        raise AttributeError(
            f"FX module target {target!r} has no attribute {name!r}"
        )
    return getattr(value, name)


def _qualified_type_name(value: Any) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _normalize_input_spec(spec: Any, name: str) -> TensorType:
    if isinstance(spec, TensorType):
        return spec
    if isinstance(spec, (tuple, list)) and all(
        isinstance(dimension, int) for dimension in spec
    ):
        return TensorType(tuple(spec))
    array = _as_numpy(spec, name)
    return TensorType(array.shape, array.dtype)


def _fetch_attr(root: Any, target: str) -> Any:
    value = root
    for atom in target.split("."):
        if not hasattr(value, atom):
            raise AttributeError(f"FX get_attr target {target!r} does not exist")
        value = getattr(value, atom)
    return value


def _as_numpy(value: Any, name: str) -> np.ndarray:
    try:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        array = np.asarray(value)
    except Exception as error:
        raise TypeError(f"FX value {name!r} cannot be converted to NumPy") from error
    if array.dtype.kind not in "biuf":
        raise TypeError(
            f"FX value {name!r} must be a numeric tensor or scalar, got "
            f"dtype {array.dtype}"
        )
    return array


def _safe_name(name: Any) -> str:
    result = re.sub(r"[^A-Za-z0-9_]", "_", str(name))
    if not result or result[0].isdigit():
        result = f"fx_{result}"
    return result


def _target_label(target: Any) -> str:
    module = getattr(target, "__module__", None)
    name = getattr(target, "__name__", None)
    if module and name:
        return f"{module}.{name}"
    return str(target)
