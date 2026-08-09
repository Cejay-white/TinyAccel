"""TinyAccel's minimal, human-readable instruction representation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

import numpy as np


class Opcode(str, Enum):
    ZERO = "ZERO"
    DMA_LOAD = "DMA_LOAD"
    MATMUL = "MATMUL"
    DMA_STORE = "DMA_STORE"


@dataclass(frozen=True)
class Instruction:
    opcode: Opcode
    operands: Mapping[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        operands = ", ".join(f"{key}={value}" for key, value in self.operands.items())
        return f"{self.opcode.value:<9} {operands}".rstrip()


@dataclass(frozen=True)
class Program:
    instructions: tuple[Instruction, ...]
    input_types: Mapping[str, tuple[tuple[int, ...], np.dtype]]
    output_name: str
    output_shape: tuple[int, ...]
    output_dtype: np.dtype

    def __str__(self) -> str:
        return "\n".join(
            f"{index:04d}: {instruction}"
            for index, instruction in enumerate(self.instructions)
        )

