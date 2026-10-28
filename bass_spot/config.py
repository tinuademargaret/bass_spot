"""Versioned, shared physical and experimental configuration."""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Physics:
    nx: int = 48
    ny: int = 48
    width: float = 6.0
    height: float = 5.0
    sound_speed: float = 343.0
    damping: float = 0.04
    source_sigma: float = 0.12
    source_clearance: float = 0.20
    reference: float = 1.0
    magnitude_floor: float = 1e-8
    frequencies: tuple[float, ...] = (60.0, 90.0, 120.0, 150.0)

    def __post_init__(self):
        if (
            min(self.nx, self.ny) < 8
            or min(
                self.width,
                self.height,
                self.sound_speed,
                self.damping,
                self.source_sigma,
                self.reference,
                self.magnitude_floor,
            )
            <= 0
        ):
            raise ValueError("Grid, physical constants and damping must be positive")
        if self.source_clearance < 0 or not self.frequencies or min(self.frequencies) <= 0:
            raise ValueError("Invalid clearance or frequencies")

    @property
    def dx(self):
        return self.width / self.nx

    @property
    def dy(self):
        return self.height / self.ny

    @property
    def shape(self):
        return self.ny, self.nx


@dataclass(frozen=True)
class Experiment:
    physics: Physics = field(default_factory=Physics)
    seed: int = 2026
    train_rooms: int = 160
    validation_rooms: int = 20
    test_rooms: int = 20
    ood_rooms: int = 20
    positions: int = 4
    epochs: int = 80
    batch_size: int = 32
    learning_rate: float = 0.001
    model_width: int = 16
    training_seeds: tuple[int, ...] = (11, 22, 33)
    wall_band: float = 0.30
    candidates: int = 20
    placement_rooms: int = 20
    # Frozen development defaults, in arbitrary simulation log-response units.
    level_threshold: float = -35.0
    quiet_weight: float = 1.0

    def __post_init__(self):
        if (
            min(
                self.train_rooms,
                self.validation_rooms,
                self.test_rooms,
                self.ood_rooms,
                self.positions,
                self.epochs,
                self.batch_size,
                self.model_width,
                self.candidates,
                self.placement_rooms,
            )
            < 1
        ):
            raise ValueError("Counts must be positive")
        if self.learning_rate <= 0 or self.wall_band <= 0 or self.quiet_weight < 0:
            raise ValueError("Invalid experiment settings")
        if not self.training_seeds:
            raise ValueError("At least one training seed is required")

    def to_dict(self):
        return asdict(self)

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:16]

    @classmethod
    def load(cls, path):
        raw = json.loads(Path(path).read_text())
        phys = raw.pop("physics", {})
        if "frequencies" in phys:
            phys["frequencies"] = tuple(phys["frequencies"])
        if "training_seeds" in raw:
            raw["training_seeds"] = tuple(raw["training_seeds"])
        return cls(physics=Physics(**phys), **raw)


def write_json(path, value):
    """Atomically replace an artifact after writing a complete JSON document."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def runtime_metadata():
    """Record the actual numerical environment used to produce an artifact."""
    import platform
    import sys

    import numpy
    import scipy
    import torch

    return dict(
        python=sys.version,
        platform=platform.platform(),
        numpy=numpy.__version__,
        scipy=scipy.__version__,
        torch=torch.__version__,
        threads=torch.get_num_threads(),
    )
