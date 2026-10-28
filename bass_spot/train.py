"""Paired training runs; validation-only checkpoint selection."""

import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from bass_spot.config import runtime_metadata, write_json
from bass_spot.data import load_split
from bass_spot.models import UNet


def masked_mae(prediction, target, mask):
    per_sample = ((prediction - target).abs() * mask).flatten(1).sum(1) / mask.flatten(1).sum(1)
    return per_sample.mean()


def train(config, data_root, output, boundary=False, seed=11, device="cpu"):
    device = torch.device(device)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Metal has no deterministic implementation for some pooling gradients.
    # Keep warnings visible and record the limitation in the run metadata.
    deterministic_warn_only = device.type == "mps"
    torch.use_deterministic_algorithms(True, warn_only=deterministic_warn_only)
    torch.set_num_threads(2)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    train_data = load_split(data_root, config, "train", boundary)
    validation = load_split(data_root, config, "validation", boundary)
    values = train_data["y"][train_data["mask"]]
    mean, scale = float(values.mean()), max(float(values.std()), 1.0)

    def loader(data, shuffle):
        dataset = TensorDataset(
            torch.from_numpy(data["x"]),
            torch.from_numpy((data["y"] - mean) / scale),
            torch.from_numpy(data["mask"].astype(np.float32)),
        )
        generator = torch.Generator().manual_seed(seed)
        return DataLoader(dataset, batch_size=config.batch_size, shuffle=shuffle, generator=generator)

    train_loader, val_loader = loader(train_data, True), loader(validation, False)
    model = UNet(4 if boundary else 3, config.model_width).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    history, best = [], float("inf")
    started = time.perf_counter()
    for epoch in range(config.epochs):
        model.train()
        train_total = 0.0
        for x, y, mask in train_loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            optimiser.zero_grad(set_to_none=True)
            loss = masked_mae(model(x), y, mask)
            loss.backward()
            optimiser.step()
            train_total += loss.item() * len(x)
        model.eval()
        val_total = 0.0
        with torch.inference_mode():
            for x, y, mask in val_loader:
                x, y, mask = x.to(device), y.to(device), mask.to(device)
                val_total += masked_mae(model(x), y, mask).item() * len(x)
        train_mae = train_total / len(train_loader.dataset) * scale
        val_mae = val_total / len(val_loader.dataset) * scale
        if not np.isfinite([train_mae, val_mae]).all():
            raise FloatingPointError("Nonfinite training or validation loss")
        history.append(dict(epoch=epoch + 1, train_mae=train_mae, validation_mae=val_mae))
        if val_mae < best:
            best = val_mae
            state = dict(
                model={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                config=config.to_dict(),
                boundary=boundary,
                seed=seed,
                mean=mean,
                scale=scale,
                best_epoch=epoch + 1,
                validation_mae=best,
            )
            temporary = output / "best.pt.tmp"
            torch.save(state, temporary)
            temporary.replace(output / "best.pt")
        write_json(
            output / "progress.json",
            dict(
                epoch=epoch + 1,
                total_epochs=config.epochs,
                train_mae=train_mae,
                validation_mae=val_mae,
                best_validation_mae=best,
                elapsed_seconds=time.perf_counter() - started,
                device=str(device),
            ),
        )
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == config.epochs:
            print(
                f"{'boundary' if boundary else 'baseline'} seed={seed} epoch={epoch + 1} "
                f"train={train_mae:.3f} validation={val_mae:.3f}",
                flush=True,
            )
    # A robust, frequency-conditioned constant baseline; training data only.
    constants = {
        str(f): float(
            np.median(
                train_data["y"][train_data["frequency"] == f][
                    train_data["mask"][train_data["frequency"] == f]
                ]
            )
        )
        for f in config.physics.frequencies
    }
    write_json(
        output / "training.json",
        dict(
            config=config.to_dict(),
            seed=seed,
            boundary=boundary,
            parameters=sum(p.numel() for p in model.parameters()),
            history=history,
            seconds=time.perf_counter() - started,
            constant_baseline=constants,
            runtime=runtime_metadata(),
            device=str(device),
            deterministic_warn_only=deterministic_warn_only,
        ),
    )
    return output / "best.pt"
