"""Held-out field errors, room-level uncertainty, and placement regret."""

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from scipy.ndimage import distance_transform_edt

from bass_spot.config import write_json
from bass_spot.data import load_split, read_manifest, read_room
from bass_spot.models import Predictor
from bass_spot.optimize import benchmark_placement, candidates_for, default_listeners


def bootstrap_mean(values, seed=2026, repetitions=2000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(len(values), size=(repetitions, len(values)))].mean(1)
    return dict(mean=float(values.mean()), ci95=np.quantile(means, [0.025, 0.975]).tolist(), n=len(values))


def region_error(error, mask):
    counts = mask.sum((1, 2, 3))
    return np.divide((error * mask).sum((1, 2, 3)), counts, out=np.full(len(error), np.nan), where=counts > 0)


def error_figure(data, prediction, scores, physics, path):
    selected = [int(np.argsort(scores)[len(scores) // 2]), int(np.argmax(scores))]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), constrained_layout=True)
    inside = data["y"][data["mask"]]
    low, high = np.quantile(inside, [0.02, 0.98])
    for row, index in enumerate(selected):
        mask = data["mask"][index, 0]
        arrays = [
            data["y"][index, 0],
            prediction[index, 0],
            np.abs(prediction[index, 0] - data["y"][index, 0]),
        ]
        for col, (title, array) in enumerate(zip(["Simulator", "Prediction", "Absolute error"], arrays)):
            kwargs = dict(vmin=low, vmax=high, cmap="viridis") if col < 2 else dict(vmin=0, cmap="magma")
            im = axes[row, col].imshow(
                np.ma.masked_where(~mask, array),
                origin="lower",
                extent=(0, physics.width, 0, physics.height),
                **kwargs,
            )
            case = "Median example" if row == 0 else "Worst example"
            axes[row, col].set_title(f"{title} · {data['frequency'][index]:g} Hz\n{case}", fontsize=10)
            axes[row, col].set_xlabel("x (m)")
            axes[row, col].set_ylabel("y (m)")
            fig.colorbar(im, ax=axes[row, col], label="relative log response (dB)")
    fig.savefig(path, dpi=140)
    plt.close(fig)


def evaluate(checkpoint, data_root, output, placements=True):
    torch.set_num_threads(2)
    predictor = Predictor(checkpoint)
    config = predictor.config
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    training = json.loads(Path(checkpoint).with_name("training.json").read_text())
    report = dict(
        checkpoint=str(checkpoint),
        boundary=predictor.boundary,
        seed=training["seed"],
        config_fingerprint=config.fingerprint,
        splits={},
    )
    for split in ["test", "ood"]:
        data = load_split(data_root, config, split, predictor.boundary)
        prediction = predictor.predict_array(data["x"])
        error = np.abs(prediction - data["y"])
        scores = region_error(error, data["mask"])
        distances = np.stack(
            [
                distance_transform_edt(np.pad(mask[0], 1), sampling=(config.physics.dy, config.physics.dx))[
                    1:-1, 1:-1
                ]
                for mask in data["mask"]
            ]
        )[:, None]
        wall = data["mask"] & (
            (distances - max(config.physics.dx, config.physics.dy) / 2) <= config.wall_band
        )
        wall_scores = region_error(error, wall)
        interior_scores = region_error(error, data["mask"] & ~wall)
        constants = np.asarray([training["constant_baseline"][str(float(f))] for f in data["frequency"]])
        constant_error = region_error(np.abs(data["y"] - constants[:, None, None, None]), data["mask"])
        rooms = []
        for room_id in np.unique(data["room"]):
            select = data["room"] == room_id
            rooms.append(
                dict(
                    room=room_id,
                    mae=float(scores[select].mean()),
                    near_wall_mae=float(wall_scores[select].mean()),
                    interior_mae=float(interior_scores[select].mean()),
                    constant_mae=float(constant_error[select].mean()),
                )
            )
        summary = dict(
            mae=bootstrap_mean([r["mae"] for r in rooms]),
            near_wall_mae=bootstrap_mean([r["near_wall_mae"] for r in rooms]),
            interior_mae=bootstrap_mean([r["interior_mae"] for r in rooms]),
            constant_mae=bootstrap_mean([r["constant_mae"] for r in rooms]),
            rooms=rooms,
            by_frequency={
                str(f): float(scores[data["frequency"] == f].mean()) for f in config.physics.frequencies
            },
            by_kind={kind: float(scores[data["kind"] == kind].mean()) for kind in np.unique(data["kind"])},
        )
        summary["worst_cases"] = [
            dict(
                room=str(data["room"][i]),
                frequency=float(data["frequency"][i]),
                kind=str(data["kind"][i]),
                mae=float(scores[i]),
            )
            for i in np.argsort(scores)[-5:][::-1]
        ]
        if placements:
            manifest = read_manifest(data_root)
            results = []
            for entry in [e for e in manifest["rooms"] if e["split"] == split][: config.placement_rooms]:
                room, arrays = read_room(data_root, entry)
                listeners = default_listeners(room, config.physics)
                # Identical candidates across model variants and training seeds.
                rng = np.random.default_rng(int(entry["geometry_hash"][:8], 16))
                candidates = candidates_for(
                    room, config.physics, listeners, config.candidates, rng, exclude=arrays["positions"]
                )
                result = benchmark_placement(room, listeners, predictor, candidates)
                result["room"] = entry["geometry_hash"]
                results.append(result)
            summary["placement"] = dict(
                rooms=results,
                regret=bootstrap_mean([r["regret"] for r in results]),
                improvement_over_random=bootstrap_mean(
                    [r["random_expected_score"] - r["selected_score"] for r in results]
                ),
                mean_solver_cold_seconds=float(np.mean([r["solver_cold_seconds"] for r in results])),
                mean_solver_warm_seconds=float(np.mean([r["solver_warm_seconds"] for r in results])),
                mean_neural_warm_seconds=float(np.mean([r["neural_warm_seconds"] for r in results])),
            )
        report["splits"][split] = summary
        error_figure(data, prediction, scores, config.physics, output / f"{split}_fields.png")
        print(
            f"{split}: MAE={summary['mae']['mean']:.3f}, constant={summary['constant_mae']['mean']:.3f}",
            flush=True,
        )
    write_json(output / "metrics.json", report)
    return report


def compare(run_root):
    """Paired room comparisons averaged over training seeds; seed variation reported separately."""
    root = Path(run_root)
    reports = [json.loads(p.read_text()) for p in sorted(root.glob("*/evaluation/metrics.json"))]
    if not reports:
        raise ValueError("No evaluation reports found")
    fingerprints = {r.get("config_fingerprint") for r in reports}
    if len(fingerprints) > 1:
        raise ValueError("Cannot compare reports from different configurations")
    comparison_figures(root, reports)
    output = {}
    lines = [
        "# Experiment results",
        "",
        "Errors are relative log-response units (dB), not calibrated SPL.",
        "",
        "| Model | Seed | Split | MAE | Constant MAE | Wall MAE | Placement regret |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for report in reports:
        for split, result in report["splits"].items():
            regret = result.get("placement", {}).get("regret", {}).get("mean")
            regret_text = f"{regret:.3f}" if regret is not None else "—"
            lines.append(
                f"| {'boundary' if report['boundary'] else 'baseline'} | {report['seed']} | {split} | "
                f"{result['mae']['mean']:.3f} | {result['constant_mae']['mean']:.3f} | "
                f"{result['near_wall_mae']['mean']:.3f} | {regret_text} |"
            )
    seeds = sorted(set(r["seed"] for r in reports))
    for split in ["test", "ood"]:
        paired = []
        for seed in seeds:
            pair = {r["boundary"]: r for r in reports if r["seed"] == seed}
            if len(pair) != 2:
                continue
            a = {r["room"]: r["mae"] for r in pair[False]["splits"][split]["rooms"]}
            b = {r["room"]: r["mae"] for r in pair[True]["splits"][split]["rooms"]}
            if a.keys() != b.keys():
                raise ValueError("Compared models have different test rooms")
            paired.append([b[room] - a[room] for room in sorted(a)])
        if paired:
            paired = np.asarray(paired)
            output[split] = dict(
                boundary_minus_baseline=bootstrap_mean(paired.mean(axis=0)),
                per_seed_mean_difference=paired.mean(axis=1).tolist(),
            )
            result = output[split]["boundary_minus_baseline"]
            lines.extend(
                [
                    "",
                    f"{split}: paired boundary − baseline MAE = {result['mean']:.3f}; "
                    f"room-bootstrap 95% CI [{result['ci95'][0]:.3f}, {result['ci95'][1]:.3f}]. "
                    "Negative values favour boundary distance. CI averages over the available seeds; "
                    "per-seed differences are recorded separately.",
                ]
            )
    lines.extend(
        [
            "",
            "Placement regret compares the recommendation with the best of the same finite candidate set.",
            "Timing includes preprocessing; solver timings separate cold assembly/factorisation and warm reuse.",
            "Training and data generation costs are separate from inference. Inspect metrics.json for details.",
        ]
    )
    write_json(root / "comparison.json", output)
    (root / "RESULTS.md").write_text("\n".join(lines) + "\n")
    return root / "RESULTS.md"


def comparison_figures(root, reports):
    """Learning curves and metrics summarised across the paired seed runs."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
    for report in reports:
        run = Path(report["checkpoint"]).parent
        training = json.loads((run / "training.json").read_text())
        history = training["history"]
        ax = axes[int(report["boundary"])]
        ax.plot(
            [h["epoch"] for h in history],
            [h["train_mae"] for h in history],
            linestyle="--",
            alpha=0.6,
            label=f"train {report['seed']}",
        )
        ax.plot(
            [h["epoch"] for h in history],
            [h["validation_mae"] for h in history],
            label=f"validation {report['seed']}",
        )
    for ax, name in zip(axes, ["Baseline", "Boundary distance"]):
        ax.set(title=name, xlabel="Epoch", ylabel="MAE (relative dB)")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7)
    fig.savefig(root / "learning_curves.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
    for ax, split in zip(axes, ["test", "ood"]):
        for boundary, name in [(False, "Baseline"), (True, "Boundary distance")]:
            selected = [r for r in reports if r["boundary"] == boundary]
            if not selected:
                continue
            frequencies = sorted(selected[0]["splits"][split]["by_frequency"], key=float)
            values = np.asarray(
                [[r["splits"][split]["by_frequency"][f] for f in frequencies] for r in selected]
            )
            ax.errorbar(
                list(map(float, frequencies)),
                values.mean(0),
                yerr=values.std(0),
                marker="o",
                capsize=3,
                label=name,
            )
        ax.set(title=split, xlabel="Frequency (Hz)", ylabel="MAE (relative dB)")
        ax.legend(fontsize=8)
    fig.suptitle("Held-out frequency error; bars show variation across training seeds")
    fig.savefig(root / "frequency_errors.png", dpi=150)
    plt.close(fig)
