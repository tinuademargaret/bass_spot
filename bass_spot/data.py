"""Deterministic room-level splits and resumable per-room dataset shards."""

import json
from pathlib import Path

import numpy as np

from bass_spot.config import write_json
from bass_spot.geometry import Room, inputs, random_room, sample_positions
from bass_spot.solver import AcousticSolver, log_response

DATA_VERSION = 1


def generate(config, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous["fingerprint"] != config.fingerprint or previous["version"] != DATA_VERSION:
            raise ValueError("Dataset configuration differs; choose another output directory")
    guard_path = output / "generation.json"
    guard = dict(version=DATA_VERSION, fingerprint=config.fingerprint)
    if guard_path.exists() and json.loads(guard_path.read_text()) != guard:
        raise ValueError("Incomplete dataset belongs to a different configuration")
    write_json(guard_path, guard)
    rng = np.random.default_rng(config.seed)
    seen, entries = set(), []
    splits = [
        ("train", config.train_rooms),
        ("validation", config.validation_rooms),
        ("test", config.test_rooms),
        ("ood", config.ood_rooms),
    ]
    for split, count in splits:
        for index in range(count):
            for _ in range(10000):
                room = random_room(config.physics, rng, ood=split == "ood")
                if room.geometry_hash not in seen:
                    break
            else:
                raise RuntimeError("Unable to sample enough unique room geometries")
            seen.add(room.geometry_hash)
            positions = sample_positions(room, config.physics, config.positions, rng)
            filename = f"{split}/{index:04d}_{room.geometry_hash}.npz"
            entry = dict(
                split=split,
                file=filename,
                geometry_hash=room.geometry_hash,
                kind=room.kind,
                parameters=room.parameters,
                positions=positions.tolist(),
            )
            entries.append(entry)
            path = output / filename
            if path.exists():
                # Check the shard is readable before treating it as complete.
                with np.load(path) as shard:
                    if shard["target"].shape != (
                        config.positions,
                        len(config.physics.frequencies),
                        *config.physics.shape,
                    ):
                        raise ValueError(f"Invalid existing shard: {path}")
                    if (
                        not np.array_equal(shard["mask"], room.mask)
                        or not np.array_equal(shard["positions"], positions)
                        or not np.isfinite(shard["target"]).all()
                    ):
                        raise ValueError(f"Existing shard contents do not match expected room: {path}")
                continue
            solver = AcousticSolver(room, config.physics)
            pressure = np.stack([solver.solve_many(positions, f) for f in config.physics.frequencies], axis=1)
            target = log_response(pressure, config.physics).astype(np.float32)
            target[:, :, ~room.mask] = 0
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            with temporary.open("wb") as handle:
                np.savez_compressed(
                    handle,
                    mask=room.mask,
                    positions=positions,
                    pressure=pressure.astype(np.complex64),
                    target=target,
                )
            temporary.replace(path)
        print(f"Generated {split}: {count} rooms", flush=True)
    manifest = dict(
        version=DATA_VERSION, fingerprint=config.fingerprint, config=config.to_dict(), rooms=entries
    )
    write_json(manifest_path, manifest)
    return manifest


def read_manifest(root):
    return json.loads((Path(root) / "manifest.json").read_text())


def read_room(root, entry):
    with np.load(Path(root) / entry["file"]) as shard:
        arrays = {key: shard[key] for key in shard.files}
    return Room(arrays["mask"], entry["kind"], entry["parameters"]), arrays


def load_split(root, config, split, boundary=False):
    manifest = read_manifest(root)
    if manifest["fingerprint"] != config.fingerprint:
        raise ValueError("Dataset does not match experiment configuration")
    xs, ys, masks, room_ids, frequencies, kinds = [], [], [], [], [], []
    for entry in manifest["rooms"]:
        if entry["split"] != split:
            continue
        room, arrays = read_room(root, entry)
        for pos_index, point in enumerate(arrays["positions"]):
            for freq_index, frequency in enumerate(config.physics.frequencies):
                xs.append(inputs(room, config.physics, point, frequency, boundary))
                ys.append(arrays["target"][pos_index, freq_index][None])
                masks.append(room.mask[None])
                room_ids.append(entry["geometry_hash"])
                frequencies.append(frequency)
                kinds.append(room.kind)
    if not xs:
        raise ValueError(f"Empty split: {split}")
    return dict(
        x=np.stack(xs),
        y=np.stack(ys),
        mask=np.stack(masks),
        room=np.asarray(room_ids),
        frequency=np.asarray(frequencies),
        kind=np.asarray(kinds),
    )
