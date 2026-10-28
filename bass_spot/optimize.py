"""Shared placement objective, feasible candidates and solver comparison."""

import time

import numpy as np
from scipy.stats import spearmanr

from bass_spot.geometry import sample_positions, valid_cells, validate_point
from bass_spot.solver import AcousticSolver, log_response


def listening_points(room, physics, centre, width=1.0, count=5):
    if count < 2 or width <= 0:
        raise ValueError("Listening region needs positive width and at least two points")
    points = np.column_stack(
        [np.linspace(centre[0] - width / 2, centre[0] + width / 2, count), np.full(count, centre[1])]
    )
    for point in points:
        validate_point(room, physics, point)
    return points


def default_listeners(room, physics):
    candidates = valid_cells(room, physics, clearance=0.35)
    centre_target = np.asarray([room.parameters["width"] * 0.45, room.parameters["height"] * 0.55])
    points = np.column_stack([(candidates[:, 1] + 0.5) * physics.dx, (candidates[:, 0] + 0.5) * physics.dy])
    for point in points[np.argsort(np.linalg.norm(points - centre_target, axis=1))]:
        try:
            return listening_points(room, physics, point)
        except ValueError:
            continue
    raise ValueError("No valid horizontal listening area fits in this room")


def candidates_for(room, physics, listeners, count, rng, exclude=()):
    # Draw a shuffled full set, then exclude near-field and previously sampled locations.
    available = sample_positions(room, physics, len(valid_cells(room, physics)), rng)
    keep = np.linalg.norm(available[:, None, :] - np.asarray(listeners)[None, :, :], axis=-1).min(1) >= 0.5
    if len(exclude):
        keep &= np.linalg.norm(available[:, None, :] - np.asarray(exclude)[None, :, :], axis=-1).min(1) > 1e-7
    available = available[keep]
    if len(available) < count:
        raise ValueError("Not enough feasible candidates away from listening points")
    return available[:count]


def score_fields(fields, listeners, room, config):
    """Lower is better. Nearest-cell sampling matches the displayed raster."""
    p = config.physics
    for point in listeners:
        validate_point(room, p, point)
    cells = np.asarray([[int(y / p.dy), int(x / p.dx)] for x, y in listeners])
    levels = fields[:, :, cells[:, 0], cells[:, 1]]  # candidate, frequency, listener
    uniformity = levels.std(axis=-1).mean(axis=-1)
    weakness = np.maximum(0, config.level_threshold - levels.mean(axis=-1)).mean(axis=-1)
    return uniformity + config.quiet_weight * weakness, uniformity, weakness


def solver_fields(solver, points):
    return np.stack(
        [log_response(solver.solve_many(points, f), solver.physics) for f in solver.physics.frequencies],
        axis=1,
    )


def recommend(room, listeners, predictor, candidates):
    fields = predictor.predict(room, candidates)
    score, uniformity, weakness = score_fields(fields, listeners, room, predictor.config)
    selected = int(np.argmin(score))
    return dict(
        index=selected,
        position=np.asarray(candidates[selected]).tolist(),
        scores=score,
        uniformity=uniformity,
        weakness=weakness,
        fields=fields,
    )


def benchmark_placement(room, listeners, predictor, candidates):
    config = predictor.config
    # Include assembly and factorisation in cold solver time.
    start = time.perf_counter()
    solver = AcousticSolver(room, config.physics)
    truth = solver_fields(solver, candidates)
    cold = time.perf_counter() - start
    start = time.perf_counter()
    solver_fields(solver, candidates)
    warm = time.perf_counter() - start
    # Warm up kernels before timing; preprocessing remains in timed predict call.
    predictor.predict(room, candidates[:1])
    start = time.perf_counter()
    prediction = recommend(room, listeners, predictor, candidates)
    neural = time.perf_counter() - start
    true_scores, uniformity, weakness = score_fields(truth, listeners, room, config)
    chosen, best = prediction["index"], int(np.argmin(true_scores))
    rank = spearmanr(prediction["scores"], true_scores).statistic
    # Fixed baseline: feasible candidate closest to lower-left corner.
    fixed = int(np.argmin(np.linalg.norm(candidates, axis=1)))
    return dict(
        selected_index=chosen,
        oracle_index=best,
        fixed_index=fixed,
        selected_score=float(true_scores[chosen]),
        oracle_score=float(true_scores[best]),
        random_expected_score=float(true_scores.mean()),
        fixed_score=float(true_scores[fixed]),
        regret=float(true_scores[chosen] - true_scores[best]),
        ranking_spearman=float(rank) if np.isfinite(rank) else None,
        uniformity=float(uniformity[chosen]),
        weakness=float(weakness[chosen]),
        solver_cold_seconds=cold,
        solver_warm_seconds=warm,
        neural_warm_seconds=neural,
        candidates=np.asarray(candidates).tolist(),
        listeners=np.asarray(listeners).tolist(),
        simulator_scores=true_scores.tolist(),
        predicted_scores=prediction["scores"].tolist(),
    )
