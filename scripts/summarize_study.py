"""Export a completed paired study and its room-level comparisons.

Run from the repository root after `bass-spot run` completes:
    python scripts/summarize_study.py --runs runs/full --output reports/full
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from bass_spot.config import Experiment, Physics, write_json
from bass_spot.evaluate import bootstrap_mean


def read(path):
    return json.loads(Path(path).read_text())


def paired_summary(reports, split, section, key):
    """Average within each room across paired seeds, then bootstrap rooms."""
    differences = []
    for seed in sorted({r["seed"] for r in reports}):
        pair = {r["boundary"]: r for r in reports if r["seed"] == seed}
        groups = []
        for boundary in [False, True]:
            result = pair[boundary]["splits"][split]
            rows = result["rooms"] if section == "field" else result["placement"]["rooms"]
            groups.append({row["room"]: row[key] for row in rows})
        if groups[0].keys() != groups[1].keys():
            raise ValueError("Paired runs have different rooms")
        differences.append([groups[1][room] - groups[0][room] for room in sorted(groups[0])])
    differences = np.asarray(differences)
    result = bootstrap_mean(differences.mean(axis=0))
    result["per_seed_mean_difference"] = differences.mean(axis=1).tolist()
    return result


def interpretation(result):
    lower, upper = result["ci95"]
    if upper < 0:
        return "favours the boundary model on these held-out rooms"
    if lower > 0:
        return "favours the baseline on these held-out rooms"
    return "does not establish a difference; the interval includes zero"


def summarize(run_root, output):
    root, output = Path(run_root), Path(output)
    experiment = read(root / "experiment.json")
    # The run file wraps the configuration alongside runtime information.
    config_dict = experiment["config"]
    config = Experiment(
        physics=Physics(**config_dict["physics"]),
        **{key: value for key, value in config_dict.items() if key != "physics"},
    )
    reports = [read(path) for path in sorted(root.glob("*/evaluation/metrics.json"))]
    expected = {(boundary, seed) for boundary in [False, True] for seed in config_dict["training_seeds"]}
    actual = [(r["boundary"], r["seed"]) for r in reports]
    if set(actual) != expected or len(actual) != len(expected):
        raise ValueError("Study is incomplete or contains duplicate model/seed runs")
    if {r["config_fingerprint"] for r in reports} != {config.fingerprint}:
        raise ValueError("Runs have different configurations")
    training = {path.parent.name: read(path) for path in root.glob("*/training.json")}
    for run in training.values():
        if run["config"] != config_dict or len(run["history"]) != config_dict["epochs"]:
            raise ValueError("Training history is incomplete or uses a different configuration")
    output.mkdir(parents=True, exist_ok=True)
    for name in [
        "RESULTS.md",
        "comparison.json",
        "learning_curves.png",
        "frequency_errors.png",
        "experiment.json",
    ]:
        shutil.copy2(root / name, output / name)
    if (root / "validation").exists():
        shutil.copytree(root / "validation", output / "validation", dirs_exist_ok=True)
    for path in sorted(root.glob("*/training.json")):
        destination = output / path.parent.name
        destination.mkdir(exist_ok=True)
        shutil.copy2(path, destination / path.name)
        for artifact in (path.parent / "evaluation").iterdir():
            shutil.copy2(artifact, destination / artifact.name)

    summary = dict(config=config_dict, models={}, paired={})
    table = []
    placement_table = []
    timing_table = []
    for boundary, name in [(False, "baseline"), (True, "boundary")]:
        selected = [r for r in reports if r["boundary"] == boundary]
        summary["models"][name] = {}
        for split in ["test", "ood"]:
            values = [r["splits"][split] for r in selected]
            means = {
                key: float(np.mean([v[key]["mean"] for v in values]))
                for key in ["mae", "near_wall_mae", "interior_mae", "constant_mae"]
            }
            placements = [v["placement"] for v in values]
            means["regret"] = float(np.mean([v["regret"]["mean"] for v in placements]))
            means["improvement_over_random"] = float(
                np.mean([v["improvement_over_random"]["mean"] for v in placements])
            )
            means["seed_mae"] = [v["mae"]["mean"] for v in values]
            means["by_frequency"] = {
                f: float(np.mean([v["by_frequency"][f] for v in values])) for f in values[0]["by_frequency"]
            }
            means["by_kind"] = {
                kind: float(np.mean([v["by_kind"][kind] for v in values])) for kind in values[0]["by_kind"]
            }
            means["ranking_spearman"] = float(
                np.mean(
                    [
                        row["ranking_spearman"]
                        for v in placements
                        for row in v["rooms"]
                        if row["ranking_spearman"] is not None
                    ]
                )
            )
            for key in ["mean_solver_cold_seconds", "mean_solver_warm_seconds", "mean_neural_warm_seconds"]:
                means[key] = float(np.mean([v[key] for v in placements]))
            summary["models"][name][split] = means
            table.append(
                f"| {name} | {split} | {means['mae']:.3f} | {means['near_wall_mae']:.3f} | "
                f"{means['interior_mae']:.3f} | {means['constant_mae']:.3f} |"
            )
            placement_table.append(
                f"| {name} | {split} | {means['regret']:.3f} | "
                f"{means['improvement_over_random']:.3f} | {means['ranking_spearman']:.3f} |"
            )
            timing_table.append(
                f"| {name} | {split} | {means['mean_solver_cold_seconds'] * 1000:.1f} | "
                f"{means['mean_solver_warm_seconds'] * 1000:.1f} | "
                f"{means['mean_neural_warm_seconds'] * 1000:.1f} |"
            )
    differences = []
    for split in ["test", "ood"]:
        summary["paired"][split] = {}
        for section, key in [("field", "mae"), ("field", "near_wall_mae"), ("placement", "regret")]:
            result = paired_summary(reports, split, section, key)
            summary["paired"][split][key] = result
            differences.append(
                f"- **{split}, {key}:** boundary − baseline = {result['mean']:.3f}; "
                f"95% room-bootstrap interval [{result['ci95'][0]:.4f}, {result['ci95'][1]:.4f}]. "
                f"This {interpretation(result)}."
            )
    summary["training_seconds"] = sum(run["seconds"] for run in training.values())
    summary["devices"] = sorted({run.get("device", "cpu") for run in training.values()})
    write_json(output / "summary.json", summary)
    baseline = summary["models"]["baseline"]
    ood_difference = summary["paired"]["ood"]["mae"]
    if ood_difference["ci95"][0] > 0:
        headline = (
            "Adding wall distance worsened average prediction error on unfamiliar L-shaped rooms. "
            "The paired room-bootstrap interval excludes zero in favour of the baseline."
        )
    elif ood_difference["ci95"][1] < 0:
        headline = (
            "Adding wall distance improved average prediction error on unfamiliar L-shaped rooms. "
            "The paired room-bootstrap interval excludes zero in favour of the boundary model."
        )
    else:
        headline = (
            "The study does not establish a generalisation benefit from wall distance: "
            "the paired OOD error interval includes zero."
        )
    findings = [headline]
    if all(baseline[split]["mae"] < baseline[split]["constant_mae"] for split in ["test", "ood"]):
        findings.append(
            "The baseline U-Net beats the training-derived constant predictor in both test groups, "
            "but retains substantial field error."
        )
    if all(
        summary["paired"][split]["regret"]["ci95"][0] <= 0 <= summary["paired"][split]["regret"]["ci95"][1]
        for split in ["test", "ood"]
    ):
        findings.append("Neither test group establishes a placement-regret difference between the models.")
    if all(
        baseline[split]["mean_neural_warm_seconds"] > baseline[split]["mean_solver_warm_seconds"]
        for split in ["test", "ood"]
    ):
        findings.append(
            "The cached numerical solver is faster than baseline neural inference on CPU "
            "for this 48 × 48 workload. These timings do not measure GPU inference."
        )
    metadata = [
        f"{name}: best epoch {min(run['history'], key=lambda h: h['validation_mae'])['epoch']}, "
        f"{run['seconds'] / 60:.1f} minutes, {run['parameters']:,} parameters"
        for name, run in sorted(training.items())
    ]
    representatives = {}
    for boundary, name in [(False, "baseline"), (True, "boundary")]:
        options = [r for r in reports if r["boundary"] == boundary]
        chosen = min(
            options,
            key=lambda r: min(
                h["validation_mae"] for h in training[Path(r["checkpoint"]).parent.name]["history"]
            ),
        )
        representatives[name] = Path(chosen["checkpoint"]).parent.name
    p = config_dict["physics"]
    total_rooms = sum(config_dict[k] for k in ["train_rooms", "validation_rooms", "test_rooms", "ood_rooms"])
    fields = total_rooms * config_dict["positions"] * len(p["frequencies"])
    lines = [
        "# Larger training study",
        "",
        f"Completed six models on {total_rooms} rooms and {fields:,} simulated fields: "
        f"{config_dict['train_rooms']} training, {config_dict['validation_rooms']} validation, "
        f"{config_dict['test_rooms']} in-distribution test and {config_dict['ood_rooms']} OOD test rooms. "
        f"Both models use width {config_dict['model_width']}, {config_dict['epochs']} epochs, "
        f"and paired seeds {config_dict['training_seeds']}.",
        "",
        "Checkpoints were selected by validation MAE. Test data did not determine training settings or stopping.",
        "",
        "## Findings",
        "",
        *findings,
        "",
        "## Field prediction",
        "",
        "Lower is better; values average the three training seeds. "
        "Units are relative log response (dB), not calibrated sound-pressure levels.",
        "",
        "| Model | Split | MAE | Near-wall MAE | Interior MAE | Constant MAE |",
        "|---|---|---:|---:|---:|---:|",
        *table,
        "",
        "## Paired comparisons",
        "",
        "Differences are averaged across seeds within each room, then bootstrapped over rooms. "
        "Negative values favour boundary distance. These intervals condition on the available training runs; "
        "they do not fully capture training uncertainty. Per-seed differences are saved in summary.json.",
        "",
        *differences,
        "",
        "The boundary input adds a small number of first-layer parameters. "
        "A positive result would still need a parameter-matched control before attributing it solely to wall distance.",
        "",
        "![Error by frequency](frequency_errors.png)",
        "",
        "## Placement",
        "",
        "Every held-out room uses 20 identical candidates across models/seeds, excluding the original "
        "dataset source positions. Regret measures the simulator score gap to the best of those candidates. "
        "Positive improvement over random is better; lower regret is better.",
        "",
        "| Model | Split | Regret | Improvement over random | Mean rank correlation |",
        "|---|---|---:|---:|---:|",
        *placement_table,
        "",
        "## CPU inference timing",
        "",
        "GPU training does not imply GPU inference: these inference measurements use two CPU threads. "
        "Cold solver includes assembly/factorisation; warm solver reuses LU factors. Neural timing includes "
        "input preparation after warmup. Timings are local measurements, not performance guarantees.",
        "",
        "| Model | Split | Cold solver (ms) | Warm solver (ms) | Neural (ms) |",
        "|---|---|---:|---:|---:|",
        *timing_table,
        "",
        "## Training and provenance",
        "",
        f"Total measured training time: {summary['training_seconds'] / 60:.1f} minutes. "
        f"Training devices: {', '.join(summary['devices'])}. "
        "Library versions and device settings are recorded in every training.json.",
        "",
        "Metal requires nondeterministic pooling-gradient operations. Seeds are fixed and this limitation "
        "is recorded as deterministic_warn_only=true; bitwise reproducibility is not claimed.",
        "",
        *[f"- {line}" for line in metadata],
        "",
        "![Training curves](learning_curves.png)",
        "",
        "## Field examples",
        "",
        "The displayed seed for each model is selected using validation error, not test error. Each figure "
        "shows its median-error and worst-error test examples with shared response colour scales.",
        "",
        f"![Baseline fields]({representatives['baseline']}/test_fields.png)",
        "",
        f"![Boundary fields]({representatives['boundary']}/test_fields.png)",
        "",
        f"![Boundary OOD fields]({representatives['boundary']}/ood_fields.png)",
        "",
        "## Scope",
        "",
        "The larger study changes training-room count, network width, training budget, library versions "
        "and device relative to the pilot; held-out rooms also differ. Comparisons with the pilot are "
        "descriptive and do not isolate the effect of additional training data. The controlled comparison "
        "here is baseline versus boundary distance within this larger study.",
        "",
        "The model remains a 2D, four-frequency, rigid-wall simulation with fixed damping and an idealised "
        "source. It is not a calibrated predictor for real rooms. No PDE residual is computed from "
        "magnitude-only predictions.",
        "",
        "[Raw per-seed results](RESULTS.md) · [Summary measurements](summary.json) · "
        "[Experiment provenance](experiment.json) · [Solver checks](validation/validation.json)",
        "",
    ]
    (output / "REPORT.md").write_text("\n".join(lines))
    return output / "REPORT.md"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default="runs/full")
    parser.add_argument("--output", default="reports/full")
    args = parser.parse_args()
    print(summarize(args.runs, args.output))


if __name__ == "__main__":
    main()
