"""Damped Helmholtz solver; e^(-i omega t), no flux at every wall face."""

import numpy as np
from scipy.sparse import coo_matrix, eye
from scipy.sparse.linalg import splu, spsolve

from bass_spot.geometry import source_map


class AcousticSolver:
    def __init__(self, room, physics):
        if room.mask.shape != physics.shape:
            raise ValueError("Mask and physical grid shapes differ")
        self.room, self.physics = room, physics
        self.cells = np.argwhere(room.mask)
        self.index = np.full(physics.shape, -1, dtype=int)
        self.index[room.mask] = np.arange(len(self.cells))
        rows, cols, values = [], [], []
        for i, (y, x) in enumerate(self.cells):
            diagonal = 0.0
            for oy, ox, weight in [
                (0, 1, physics.dx**-2),
                (0, -1, physics.dx**-2),
                (1, 0, physics.dy**-2),
                (-1, 0, physics.dy**-2),
            ]:
                yy, xx = y + oy, x + ox
                if 0 <= yy < physics.ny and 0 <= xx < physics.nx and room.mask[yy, xx]:
                    rows.append(i)
                    cols.append(self.index[yy, xx])
                    values.append(weight)
                    diagonal -= weight
            rows.append(i)
            cols.append(i)
            values.append(diagonal)
        n = len(self.cells)
        self.laplacian = coo_matrix((values, (rows, cols)), shape=(n, n)).tocsc()
        self._factors = {}

    def matrix(self, frequency):
        if not np.isfinite(frequency) or frequency <= 0:
            raise ValueError("Frequency must be positive")
        p = self.physics
        k = 2 * np.pi * frequency / p.sound_speed
        return (self.laplacian + k * k * (1 + 1j * p.damping) * eye(len(self.cells), format="csc")).tocsc()

    def factor(self, frequency):
        if frequency not in self._factors:
            self._factors[frequency] = splu(self.matrix(frequency))
        return self._factors[frequency]

    def solve_many(self, points, frequency, reuse=True):
        points = list(points)
        if not points:
            raise ValueError("At least one source is required")
        rhs = np.column_stack([-source_map(self.room, self.physics, p)[self.room.mask] for p in points])
        solution = self.factor(frequency).solve(rhs) if reuse else spsolve(self.matrix(frequency), rhs)
        solution = solution.reshape(len(self.cells), len(points))
        if not np.isfinite(solution).all():
            raise FloatingPointError("Nonfinite pressure solution")
        fields = np.zeros((len(points), *self.physics.shape), dtype=np.complex128)
        fields[:, self.room.mask] = solution.T
        return fields

    def solve(self, point, frequency, reuse=True):
        return self.solve_many([point], frequency, reuse)[0]


def log_response(pressure, physics):
    return 20 * np.log10(np.maximum(np.abs(pressure), physics.magnitude_floor) / physics.reference)
