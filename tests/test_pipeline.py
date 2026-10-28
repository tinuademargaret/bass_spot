from dataclasses import replace

import numpy as np
import pytest
import torch

from bass_spot.config import Experiment, Physics
from bass_spot.data import generate, load_split, read_room
from bass_spot.geometry import make_room
from bass_spot.models import Predictor, UNet
from bass_spot.optimize import candidates_for, default_listeners, score_fields
from bass_spot.train import masked_mae, train


def tiny_config():
    return Experiment(
        physics=Physics(nx=16, ny=16),
        train_rooms=2,
        validation_rooms=1,
        test_rooms=1,
        ood_rooms=1,
        positions=2,
        epochs=2,
        model_width=2,
        batch_size=8,
        training_seeds=(11,),
        candidates=3,
        placement_rooms=1,
    )


def test_generation_resume_and_checkpoint(tmp_path, monkeypatch):
    config = tiny_config()
    root = tmp_path / "data"
    manifest = generate(config, root)
    hashes = [r["geometry_hash"] for r in manifest["rooms"]]
    assert len(set(hashes)) == len(hashes)
    assert manifest == generate(config, root)
    shard = root / manifest["rooms"][0]["file"]
    before = shard.stat().st_mtime_ns
    generate(config, root)
    assert shard.stat().st_mtime_ns == before
    with pytest.raises(ValueError):
        generate(replace(config, seed=123), root)
    dataset = load_split(root, config, "train", boundary=True)
    assert dataset["x"].shape == (16, 4, 16, 16)
    assert np.isfinite(dataset["y"]).all()
    checkpoint = train(config, root, tmp_path / "model")
    predictor = Predictor(checkpoint)
    room, arrays = read_room(root, manifest["rooms"][0])
    prediction = predictor.predict(room, arrays["positions"])
    assert prediction.shape == (2, 4, 16, 16)
    assert np.isfinite(prediction).all()
    assert np.all(prediction[:, :, ~room.mask] == 0)

    from bass_spot.evaluate import compare, evaluate

    metrics = evaluate(checkpoint, root, tmp_path / "model" / "evaluation")
    assert metrics["splits"]["test"]["mae"]["n"] == 1
    assert metrics["splits"]["ood"]["placement"]["regret"]["mean"] >= 0
    assert compare(tmp_path).exists()

    # Exercise the live interface with a real checkpoint, not mocked predictions.
    import shutil
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    app_path = Path(__file__).resolve().parents[1] / "app.py"
    model_directory = tmp_path / "runs" / "test" / "baseline_11"
    model_directory.mkdir(parents=True)
    shutil.copy2(checkpoint, model_directory / "best.pt")
    monkeypatch.chdir(tmp_path)
    app = AppTest.from_file(str(app_path), default_timeout=30).run()
    assert not app.exception
    assert app.button[0].disabled
    app.selectbox[0].select("runs/test/baseline_11/best.pt").run()
    assert not app.exception
    assert not app.button[0].disabled
    app.button[0].click().run()
    assert not app.exception
    assert app.success
    assert len(app.metric) == 3
    # Moving the speaker must clear a stale recommendation.
    next(s for s in app.slider if s.label == "Speaker x (m)").set_value(1.2).run()
    assert not app.success
    app.selectbox[1].select("L-shaped").run()
    assert not app.exception
    next(s for s in app.slider if s.label == "Speaker x (m)").set_value(4.5).run()
    next(s for s in app.slider if s.label == "Speaker y (m)").set_value(3.5).run()
    assert app.error
    assert not app.exception


def test_masked_loss_ignores_exterior_and_weights_samples_equally():
    prediction = torch.tensor([[[[2.0, 100.0]]], [[[4.0, 4.0]]]])
    target = torch.zeros_like(prediction)
    mask = torch.tensor([[[[1.0, 0.0]]], [[[1.0, 1.0]]]])
    assert masked_mae(prediction, target, mask).item() == 3


def test_placement_objective_and_constraints():
    config = Experiment()
    room = make_room(config.physics, 5, 4, 2, 1)
    listeners = default_listeners(room, config.physics)
    candidates = candidates_for(room, config.physics, listeners, 20, np.random.default_rng(4))
    assert np.linalg.norm(candidates[:, None] - listeners[None], axis=-1).min() >= 0.5
    fields = np.full((2, 4, 48, 48), -20.0)
    fields[1] = -60
    scores, uniformity, weakness = score_fields(fields, listeners, room, config)
    assert scores[0] < scores[1]
    assert np.all(uniformity == 0)
    assert weakness[1] > 0


def test_model_can_overfit_one_simulated_field():
    from bass_spot.geometry import inputs
    from bass_spot.solver import AcousticSolver, log_response

    torch.set_num_threads(2)
    torch.manual_seed(8)
    physics = Physics(nx=16, ny=16)
    room = make_room(physics)
    field = log_response(AcousticSolver(room, physics).solve((1, 1), 90), physics)
    target = torch.tensor(field[None, None], dtype=torch.float32)
    mask = torch.tensor(room.mask[None, None], dtype=torch.float32)
    inside = target[mask.bool()]
    mean, scale = inside.mean(), inside.std()
    target = (target - mean) / scale
    x = torch.tensor(inputs(room, physics, (1, 1), 90)[None])
    model = UNet(3, 4)
    optimiser = torch.optim.Adam(model.parameters(), lr=0.005)
    initial = masked_mae(model(x), target, mask).item()
    for step in range(240):
        if step == 120:
            optimiser.param_groups[0]["lr"] = 0.001
        optimiser.zero_grad()
        loss = masked_mae(model(x), target, mask)
        loss.backward()
        optimiser.step()
    assert masked_mae(model(x), target, mask).item() < initial * 0.4
