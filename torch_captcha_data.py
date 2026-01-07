from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


@dataclass(frozen=True)
class ImagePreprocess:
    image_height: int = 100
    image_width: int = 120


def pil_to_tensor_rgb(image: Image.Image, preprocess: ImagePreprocess) -> torch.Tensor:
    # Explicitly resize via PIL (not np.resize) to avoid data corruption.
    img = image.convert("RGB").resize((preprocess.image_width, preprocess.image_height))
    arr = np.asarray(img, dtype=np.float32) / 255.0  # (H, W, 3)
    tensor = torch.from_numpy(arr).permute(2, 0, 1)  # (3, H, W)
    return tensor


class CsvCaptchaDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        csv_path: str | Path,
        images_dir: str | Path,
        letterstr: str,
        captcha_len: int = 4,
        preprocess: ImagePreprocess | None = None,
    ):
        self.csv_path = Path(csv_path)
        self.images_dir = Path(images_dir)
        self.letterstr = letterstr
        self.captcha_len = captcha_len
        self.preprocess = preprocess or ImagePreprocess()

        self.samples: List[Tuple[str, str]] = []
        with self.csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                if len(row) < 2:
                    raise ValueError(f"Bad CSV row in {self.csv_path}: {row}")
                image_id, label = row[0].strip(), row[1].strip()
                if len(label) != self.captcha_len:
                    raise ValueError(
                        f"Label length mismatch: expected {self.captcha_len}, got {len(label)} ({label!r})"
                    )
                self.samples.append((image_id, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_id, label = self.samples[index]
        image_path = self.images_dir / f"{image_id}.png"
        with Image.open(image_path) as img:
            x = pil_to_tensor_rgb(img, self.preprocess)

        # Return class indices for each char (shape: (captcha_len,)).
        y_indices = []
        for ch in label:
            idx = self.letterstr.find(ch)
            if idx < 0:
                raise ValueError(f"Unsupported character: {ch!r} in label {label!r}")
            y_indices.append(idx)
        y = torch.tensor(y_indices, dtype=torch.long)
        return x, y
