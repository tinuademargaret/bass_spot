import numpy as np
import pytest
from scipy.ndimage import label

from bass_spot.config import Physics
from bass_spot.geometry import distance_map, make_room, random_room, sample_positions, source_map
from bass_spot.solver import AcousticSolver
from bass_spot.validation import modal_check, refinement_check


@pytest.mark.parametrize("cut", [False, True])
def test_no_flux_and_residual(cut):
    physics = Physics(nx=24, ny=24)
    room = make_room(physics, 5, 4, 2 if cut else 0, 1.5 if cut else 0)
    solver = AcousticSolver(room, physics)
    laplacian = solver.laplacian
    np.testing.assert_allclose(laplacian @ np.ones(room.mask.sum()), 0, atol=1e-12)
    np.testing.assert_allclose((laplacian - laplacian.T).toarray(), 0)
    pressure = solver.solve((1, 1), 90)
    source = source_map(room, physics, (1, 1))
    np.testing.assert_allclose(solver.matrix(90) @ pressure[room.mask], -source[room.mask], atol=1e-11)
    assert np.all(pressure[~room.mask] == 0)
    direct = solver.solve((1, 1), 90, reuse=False)
    np.testing.assert_allclose(pressure, direct, atol=1e-11)


def test_source_strength_and_reciprocity():
    physics = Physics()
    room = make_room(physics)
    solver = AcousticSolver(room, physics)
    a, b = (1, 1), (3, 2)
    sa, sb = source_map(room, physics, a), source_map(room, physics, b)
    assert sa.sum() * physics.dx * physics.dy == pytest.approx(1)
    pa, pb = solver.solve_many([a, b], 90)
    assert np.sum(sb * pa) == pytest.approx(np.sum(sa * pb), abs=1e-10)
    assert not np.allclose(pa, pb)
    assert not np.allclose(pa, solver.solve(a, 120))
    assert len(solver._factors) == 2


def test_modes_and_grid_convergence():
    physics = Physics()
    assert modal_check(physics)["maximum_relative_error"] < 0.02
    for kind in ["rectangle", "L"]:
        assert refinement_check(physics, kind)["relative_complex_l2"] < 0.15


def test_geometry_and_distance():
    physics = Physics()
    rng = np.random.default_rng(13)
    for _ in range(20):
        room = random_room(physics, rng)
        assert label(room.mask)[1] == 1
        assert np.all(distance_map(room, physics)[~room.mask] == 0)
        positions = sample_positions(room, physics, 4, rng)
        assert len(np.unique(positions, axis=0)) == 4
    full = make_room(physics, 6, 5)
    distances = distance_map(full, physics)
    assert distances[0, 20] == pytest.approx(physics.dy)
    assert distances[-1, 20] == pytest.approx(physics.dy)
    assert distances[20, 0] == pytest.approx(physics.dx)
    with pytest.raises(ValueError):
        source_map(make_room(physics), physics, (5.9, 4.9))
    with pytest.raises(ValueError):
        make_room(physics, 5, 4, 5, 2)


def test_reflections_have_same_hash():
    from bass_spot.geometry import Room

    room = make_room(Physics(), 5, 4, 2, 1)
    assert room.geometry_hash == Room(room.mask[:, ::-1], "L", {}).geometry_hash
