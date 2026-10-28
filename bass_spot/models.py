"""Small U-Net and checkpoint-backed batched inference."""

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from bass_spot.config import Experiment, Physics
from bass_spot.geometry import inputs


class Block(nn.Sequential):
    def __init__(self, inputs, outputs):
        super().__init__(
            nn.Conv2d(inputs, outputs, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(outputs, outputs, 3, padding=1),
            nn.ReLU(),
        )


class UNet(nn.Module):
    def __init__(self, channels=3, width=16):
        super().__init__()
        self.encoders = nn.ModuleList(
            [Block(channels, width), Block(width, width * 2), Block(width * 2, width * 4)]
        )
        self.bottleneck = Block(width * 4, width * 8)
        self.decoders = nn.ModuleList(
            [Block(width * 12, width * 4), Block(width * 6, width * 2), Block(width * 3, width)]
        )
        self.output = nn.Conv2d(width, 1, 1)

    def forward(self, x):
        skips = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.bottleneck(x)
        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = decoder(torch.cat([x, skip], dim=1))
        return self.output(x)


class Predictor:
    def __init__(self, checkpoint, device="cpu"):
        state = torch.load(Path(checkpoint), map_location=device, weights_only=True)
        raw = dict(state["config"])
        raw["physics"] = Physics(**raw["physics"])
        self.config = Experiment(**raw)
        self.boundary = state["boundary"]
        self.mean, self.scale = state["mean"], state["scale"]
        self.device = torch.device(device)
        self.model = UNet(4 if self.boundary else 3, self.config.model_width).to(self.device)
        self.model.load_state_dict(state["model"])
        self.model.eval()

    def predict_array(self, x, batch_size=64):
        results = []
        with torch.inference_mode():
            for start in range(0, len(x), batch_size):
                batch = torch.as_tensor(
                    x[start : start + batch_size], dtype=torch.float32, device=self.device
                )
                results.append((self.model(batch) * self.scale + self.mean).cpu().numpy())
        return np.concatenate(results)

    def predict(self, room, points, frequencies=None):
        frequencies = self.config.physics.frequencies if frequencies is None else frequencies
        points = list(points)
        x = np.stack(
            [
                inputs(room, self.config.physics, point, f, self.boundary)
                for point in points
                for f in frequencies
            ]
        )
        result = self.predict_array(x).reshape(len(points), len(frequencies), *room.mask.shape)
        result[:, :, ~room.mask] = 0
        return result
