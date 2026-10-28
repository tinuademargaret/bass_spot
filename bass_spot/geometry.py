"""Cell-centred, axis-aligned rooms with boundaries on cell faces."""

import hashlib
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt, label

from bass_spot.config import Physics


@dataclass
class Room:
    mask: np.ndarray
    kind: str
    parameters: dict

    def __post_init__(self):
        self.mask = np.asarray(self.mask, dtype=bool)
        if self.mask.ndim != 2 or not self.mask.any() or label(self.mask)[1] != 1:
            raise ValueError("Room must be a nonempty, connected 2-D mask")

    @property
    def geometry_hash(self):
        # Canonicalise reflections and half-turns so equivalent rooms stay together.
        variants = [self.mask, self.mask[::-1], self.mask[:, ::-1], self.mask[::-1, ::-1]]
        payload = min(np.packbits(a).tobytes() for a in variants)
        return hashlib.sha256(str(self.mask.shape).encode() + payload).hexdigest()[:20]


def make_room(physics: Physics, width=5.0, height=4.0, cut_width=0.0, cut_height=0.0):
    if not (0 < width <= physics.width and 0 < height <= physics.height):
        raise ValueError("Room dimensions must fit inside the canvas")
    if min(cut_width, cut_height) < 0 or (cut_width == 0) != (cut_height == 0):
        raise ValueError("Both cutout dimensions must be positive, or both zero")
    nx, ny = round(width / physics.dx), round(height / physics.dy)
    cx, cy = round(cut_width / physics.dx), round(cut_height / physics.dy)
    if nx < 4 or ny < 4 or cx >= nx or cy >= ny:
        raise ValueError("Room or cutout dimensions leave insufficient air cells")
    if cut_width > 0 and (cx < 1 or cy < 1):
        raise ValueError("Cutout must occupy at least one cell in each axis")
    mask = np.zeros(physics.shape, dtype=bool)
    mask[:ny, :nx] = True
    if cx and cy:
        mask[ny - cy : ny, nx - cx : nx] = False
    return Room(
        mask,
        "L" if cx and cy else "rectangle",
        dict(
            width=nx * physics.dx,
            height=ny * physics.dy,
            cut_width=cx * physics.dx,
            cut_height=cy * physics.dy,
        ),
    )


def random_room(physics, rng, ood=False):
    width = rng.uniform(3.5, physics.width)
    height = rng.uniform(3.0, physics.height)
    is_l = ood or rng.random() < 0.5
    low, high = (0.55, 0.72) if ood else (0.20, 0.45)
    cw = width * rng.uniform(low, high) if is_l else 0
    ch = height * rng.uniform(low, high) if is_l else 0
    return make_room(physics, width, height, cw, ch)


def distance_map(room, physics):
    """Distance to exterior cell centres (not exact wall-face distance), in metres."""
    padded = np.pad(room.mask, 1, constant_values=False)
    return distance_transform_edt(padded, sampling=(physics.dy, physics.dx))[1:-1, 1:-1].astype(np.float32)


def coordinates(physics):
    return np.meshgrid((np.arange(physics.nx) + 0.5) * physics.dx, (np.arange(physics.ny) + 0.5) * physics.dy)


def valid_cells(room, physics, clearance=None):
    clearance = physics.source_clearance if clearance is None else clearance
    # Conservative conversion from exterior-centre distance to clearance.
    distance = distance_map(room, physics) - max(physics.dx, physics.dy) / 2
    return np.argwhere(room.mask & (distance >= clearance))


def sample_positions(room, physics, count, rng, exclude=()):
    cells = valid_cells(room, physics)
    excluded = {tuple(np.round(p, 8)) for p in exclude}
    points = [((x + 0.5) * physics.dx, (y + 0.5) * physics.dy) for y, x in cells]
    points = [p for p in points if tuple(np.round(p, 8)) not in excluded]
    if len(points) < count:
        raise ValueError("Not enough distinct valid speaker positions")
    return np.asarray(points)[rng.choice(len(points), count, replace=False)]


def validate_point(room, physics, point):
    x, y = map(float, point)
    if not np.isfinite([x, y]).all() or not (0 <= x < physics.width and 0 <= y < physics.height):
        raise ValueError("Point lies outside the canvas")
    if not room.mask[int(y / physics.dy), int(x / physics.dx)]:
        raise ValueError("Point lies outside the room")


def source_map(room, physics, point):
    validate_point(room, physics, point)
    x, y = coordinates(physics)
    source = np.exp(-((x - point[0]) ** 2 + (y - point[1]) ** 2) / (2 * physics.source_sigma**2)) * room.mask
    total = source.sum() * physics.dx * physics.dy
    if total <= 0:
        raise ValueError("Source is not resolved on this grid")
    return source / total


def inputs(room, physics, point, frequency, boundary=False):
    if frequency not in physics.frequencies:
        raise ValueError("Model supports only the configured discrete frequencies")
    channels = [
        room.mask.astype(np.float32),
        source_map(room, physics, point).astype(np.float32),
        np.full(physics.shape, frequency / max(physics.frequencies), dtype=np.float32),
    ]
    if boundary:
        channels.append(distance_map(room, physics) / max(physics.width, physics.height))
    return np.stack(channels)
