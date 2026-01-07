from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader

from torch_captcha_data import CsvCaptchaDataset, ImagePreprocess
from torch_captcha_model import CaptchaCNN, CaptchaConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained PyTorch captcha model")
    p.add_argument("--csv", default="./data/manual_label/captcha_test.csv")
    p.add_argument("--dir", default="./data/manual_label/")
    p.add_argument("--ckpt", default="./data/model/imitate_4_model_torch.pt")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--limit", type=int, default=0, help="0 means no limit")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cpu")

    ckpt = torch.load(args.ckpt, map_location=device)
    config = CaptchaConfig(**ckpt.get("config", {}))
    preprocess = ImagePreprocess(image_height=config.image_height, image_width=config.image_width)

    ds = CsvCaptchaDataset(
        csv_path=args.csv,
        images_dir=args.dir,
        letterstr=config.letterstr,
        captcha_len=config.captcha_len,
        preprocess=preprocess,
    )
    if args.limit and args.limit > 0:
        ds.samples = ds.samples[: args.limit]

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    model = CaptchaCNN(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    total = 0
    correct_full = 0
    correct_digit = [0 for _ in range(config.captcha_len)]

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            preds_text = model.predict_text(x)

            # Build ground truth text for printing/metrics
            for i in range(x.shape[0]):
                gt = "".join(config.letterstr[int(y[i, j].item())] for j in range(config.captcha_len))
                pred = preds_text[i]
                total += 1
                if pred == gt:
                    correct_full += 1
                for j in range(config.captcha_len):
                    if pred[j] == gt[j]:
                        correct_digit[j] += 1

                print(f"AI解析:{pred} 答案:{gt}")

    if total == 0:
        print("No samples.")
        return

    print(f"full acc:{correct_full/total*100:.2f}%")
    for i in range(config.captcha_len):
        print(f"digit{i+1} acc:{correct_digit[i]/total*100:.2f}%")


if __name__ == "__main__":
    main()
