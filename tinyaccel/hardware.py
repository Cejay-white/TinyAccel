"""Configuration for the teaching-oriented TinyAccel device."""

from dataclasses import dataclass


@dataclass(frozen=True)
class HardwareConfig:
    """Simple throughput and memory model used by the simulator."""

    sram_bytes: int = 256 * 1024
    dma_bytes_per_cycle: int = 32
    macs_per_cycle: int = 64

    def __post_init__(self) -> None:
        for name, value in (
            ("sram_bytes", self.sram_bytes),
            ("dma_bytes_per_cycle", self.dma_bytes_per_cycle),
            ("macs_per_cycle", self.macs_per_cycle),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")

