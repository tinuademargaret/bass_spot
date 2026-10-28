"""Small command-line surface for reproducible experiments."""

import argparse
import time
from pathlib import Path

from bass_spot.config import Experiment, runtime_metadata, write_json


def main():
    parser = argparse.ArgumentParser(description="The Bass Spot acoustic surrogate experiment")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser(
        "validate", help="Run numerical validation and generate diagnostic figures"
    )
    validate.add_argument("--output", default="reports/validation")
    for name in ["generate", "train", "evaluate", "run"]:
        sub = commands.add_parser(name)
        if name != "evaluate":
            sub.add_argument("--config", default="configs/full.json")
        sub.add_argument("--data", default="data/full")
        if name in ["train", "run"]:
            sub.add_argument("--device", default="cpu")
        if name == "generate":
            continue
        sub.add_argument("--output", default="runs/full")
        if name == "train":
            sub.add_argument("--boundary", action="store_true")
            sub.add_argument("--seed", type=int, default=11)
        if name == "evaluate":
            sub.add_argument("--checkpoint", required=True)
            sub.add_argument("--skip-placement", action="store_true")
    sub = commands.add_parser("compare")
    sub.add_argument("--runs", default="runs/full")
    args = parser.parse_args()
    if args.command == "validate":
        from bass_spot.validation import validate

        validate(args.output)
    elif args.command == "generate":
        from bass_spot.data import generate

        generate(Experiment.load(args.config), args.data)
    elif args.command == "train":
        from bass_spot.train import train

        train(Experiment.load(args.config), args.data, args.output, args.boundary, args.seed, args.device)
    elif args.command == "evaluate":
        from bass_spot.evaluate import evaluate

        evaluate(args.checkpoint, args.data, args.output, not args.skip_placement)
    elif args.command == "compare":
        from bass_spot.evaluate import compare

        print(compare(args.runs))
    elif args.command == "run":
        from bass_spot.data import generate
        from bass_spot.evaluate import compare, evaluate
        from bass_spot.train import train
        from bass_spot.validation import validate

        config = Experiment.load(args.config)
        output = Path(args.output)
        validate(output / "validation")
        start = time.perf_counter()
        generate(config, args.data)
        generation_seconds = time.perf_counter() - start
        for boundary in [False, True]:
            for seed in config.training_seeds:
                run = output / f"{'boundary' if boundary else 'baseline'}_{seed}"
                checkpoint = train(config, args.data, run, boundary, seed, args.device)
                evaluate(checkpoint, args.data, run / "evaluation")
        write_json(
            output / "experiment.json",
            dict(
                config=config.to_dict(),
                generation_seconds=generation_seconds,
                runtime=runtime_metadata(),
                timing_note="Generation time can include resumed shards",
            ),
        )
        print(compare(output))


if __name__ == "__main__":
    main()
