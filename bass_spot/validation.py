"""Independent modal, refinement, damping and algebraic solver checks."""

from dataclasses import replace
from pathlib import Path

import matplotlib
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse.linalg import eigsh

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bass_spot.config import Physics, write_json
from bass_spot.geometry import coordinates, make_room, random_room, sample_positions, source_map
from bass_spot.solver import AcousticSolver, log_response


def modal_check(physics, count=9):
    room = make_room(physics, 5.0, 4.0)
    solver = AcousticSolver(room, physics)
    eigenvalues = np.sort(eigsh(-solver.laplacian, k=count + 1, which="SM", return_eigenvectors=False))[1:]
    numerical = physics.sound_speed * np.sqrt(eigenvalues) / (2 * np.pi)
    w, h = room.parameters["width"], room.parameters["height"]
    analytical = sorted(
        physics.sound_speed / 2 * np.sqrt((m / w) ** 2 + (n / h) ** 2)
        for m in range(6)
        for n in range(6)
        if m + n
    )[:count]
    relative = np.abs(numerical - analytical) / analytical
    return dict(
        analytical_hz=analytical,
        numerical_hz=numerical.tolist(),
        maximum_relative_error=float(relative.max()),
    )


def refinement_check(physics, kind="rectangle", frequency=73.0):
    dimensions = dict(
        width=5.0, height=3.75, cut_width=2.0 if kind == "L" else 0, cut_height=1.25 if kind == "L" else 0
    )
    fine_physics = replace(physics, nx=physics.nx * 2, ny=physics.ny * 2)
    coarse_room, fine_room = make_room(physics, **dimensions), make_room(fine_physics, **dimensions)
    point = (1.3, 1.1)
    coarse = AcousticSolver(coarse_room, physics).solve(point, frequency)
    fine = AcousticSolver(fine_room, fine_physics).solve(point, frequency)
    x, y = coordinates(physics)
    interpolator = RegularGridInterpolator(
        (
            (np.arange(fine_physics.ny) + 0.5) * fine_physics.dy,
            (np.arange(fine_physics.nx) + 0.5) * fine_physics.dx,
        ),
        fine,
    )
    resampled = interpolator(np.column_stack([y.ravel(), x.ravel()])).reshape(physics.shape)
    select = coarse_room.mask & ((x - point[0]) ** 2 + (y - point[1]) ** 2 > 0.4**2)
    error = np.linalg.norm((coarse - resampled)[select]) / np.linalg.norm(resampled[select])
    return dict(relative_complex_l2=float(error), frequency=frequency, excluded_source_radius_m=0.4)


def validate(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    physics = Physics()
    room = make_room(physics, 5, 4)
    solver = AcousticSolver(room, physics)
    point, frequency = (1.3, 1.1), 90
    pressure = solver.solve(point, frequency)
    rhs = -source_map(room, physics, point)[room.mask]
    residual = np.linalg.norm(solver.matrix(frequency) @ pressure[room.mask] - rhs) / np.linalg.norm(rhs)
    constant_error = np.abs(solver.laplacian @ np.ones(room.mask.sum())).max()
    modal = modal_check(physics)
    refinements = {kind: refinement_check(physics, kind) for kind in ["rectangle", "L"]}
    bass_refinements = {
        kind: {str(f): refinement_check(physics, kind, f) for f in physics.frequencies}
        for kind in ["rectangle", "L"]
    }
    # Frequency sweep excites modes from an off-centre, near-corner source.
    frequencies = np.linspace(25, 165, 281)
    response = [
        float(np.sqrt(np.mean(np.abs(solver.solve((0.4, 0.4), f, reuse=False)[room.mask]) ** 2)))
        for f in frequencies
    ]
    fig, ax = plt.subplots(figsize=(8, 3), constrained_layout=True)
    ax.plot(frequencies, 20 * np.log10(response))
    for f in modal["analytical_hz"]:
        if frequencies[0] <= f <= frequencies[-1]:
            ax.axvline(f, color="grey", alpha=0.35, linewidth=0.8)
    ax.set(
        xlabel="Frequency (Hz)",
        ylabel="Room RMS response (relative dB)",
        title="Rigid rectangle: frequency response and analytical modes",
    )
    fig.savefig(output / "resonances.png", dpi=140)
    plt.close(fig)
    rng = np.random.default_rng(2026)
    fig, axes = plt.subplots(3, 4, figsize=(10, 7), constrained_layout=True)
    for ax in axes.flat:
        generated = random_room(physics, rng)
        positions = sample_positions(generated, physics, 4, rng)
        ax.imshow(generated.mask, origin="lower", extent=(0, physics.width, 0, physics.height), cmap="Greys")
        ax.scatter(positions[:, 0], positions[:, 1], s=12, c="tomato")
        ax.set_title(generated.kind)
    fig.savefig(output / "geometry_gallery.png", dpi=140)
    plt.close(fig)
    damping = {}
    for loss in [0.02, 0.04, 0.08]:
        p = replace(physics, damping=loss)
        field = AcousticSolver(room, p).solve(point, 90)
        damping[str(loss)] = dict(
            mean_log_response=float(log_response(field, p)[room.mask].mean()),
            maximum_magnitude=float(np.abs(field[room.mask]).max()),
        )
    passed = bool(
        residual < 1e-10
        and constant_error < 1e-10
        and modal["maximum_relative_error"] < 0.02
        and all(r["relative_complex_l2"] < 0.15 for r in refinements.values())
        and all(
            r["relative_complex_l2"] < 0.15 for group in bass_refinements.values() for r in group.values()
        )
    )
    result = dict(
        passed=passed,
        relative_residual=float(residual),
        constant_laplacian_error=float(constant_error),
        modes=modal,
        refinements=refinements,
        bass_frequency_refinements=bass_refinements,
        damping_sensitivity=damping,
        note="Refinement thresholds apply to these fixtures, not every possible room/frequency.",
    )
    write_json(output / "validation.json", result)
    if not passed:
        raise AssertionError(f"Solver validation failed: {result}")
    print(
        f"Solver validation passed: residual={residual:.2e}, mode error={modal['maximum_relative_error']:.4f}",
        flush=True,
    )
    return result
