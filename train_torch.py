from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from torch_captcha_data import CsvCaptchaDataset, ImagePreprocess
from torch_captcha_model import CaptchaCNN, CaptchaConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train multi-head CNN (PyTorch, CPU-only by default)")
    p.add_argument("--train-csv", default="./data/4_imitate_train_set/captcha_train.csv")
    p.add_argument("--train-dir", default="./data/4_imitate_train_set/")
    p.add_argument("--val-csv", default="./data/4_imitate_vali_set/captcha_vali.csv")
    p.add_argument("--val-dir", default="./data/4_imitate_vali_set/")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="./data/model/imitate_4_model_torch.pt")
    return p.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)


def compute_metrics(logits_list, y: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
    # logits_list: list[(N, C)], y: (N, L)
    batch = y.shape[0]
    captcha_len = y.shape[1]

    correct_each = []
    preds_each = []
    for i in range(captcha_len):
        preds = torch.argmax(logits_list[i], dim=1)
        preds_each.append(preds)
        correct_each.append((preds == y[:, i]).float())

    correct_each_tensor = torch.stack(correct_each, dim=1)  # (N, L)
    full_correct = (correct_each_tensor.sum(dim=1) == captcha_len).float().mean()

    mean_digit_acc = correct_each_tensor.mean(dim=1).mean()

    metrics: Dict[str, float] = {
        "acc_full": float(full_correct.item()),
        "acc_mean_digit": float(mean_digit_acc.item()),
    }
    for i in range(captcha_len):
        metrics[f"acc_digit{i+1}"] = float(correct_each_tensor[:, i].mean().item())

    return full_correct, metrics


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = CaptchaConfig()
    preprocess = ImagePreprocess(image_height=config.image_height, image_width=config.image_width)

    train_ds = CsvCaptchaDataset(
        csv_path=args.train_csv,
        images_dir=args.train_dir,
        letterstr=config.letterstr,
        captcha_len=config.captcha_len,
        preprocess=preprocess,
    )
    val_ds = CsvCaptchaDataset(
        csv_path=args.val_csv,
        images_dir=args.val_dir,
        letterstr=config.letterstr,
        captcha_len=config.captcha_len,
        preprocess=preprocess,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = CaptchaCNN(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_val_full = -1.0
    best_val_mean_digit = -1.0
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits_list = model(x)
            losses = [criterion(logits_list[i], y[:, i]) for i in range(config.captcha_len)]
            loss = sum(losses) / config.captcha_len
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * x.shape[0]

        train_loss = running_loss / max(1, len(train_ds))

        model.eval()
        with torch.no_grad():
            val_loss_sum = 0.0
            metric_sum = {"acc_full": 0.0, "acc_mean_digit": 0.0}
            for i in range(config.captcha_len):
                metric_sum[f"acc_digit{i+1}"] = 0.0

            n = 0
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                logits_list = model(x)
                losses = [criterion(logits_list[i], y[:, i]) for i in range(config.captcha_len)]
                loss = sum(losses) / config.captcha_len

                _, metrics = compute_metrics(logits_list, y)
                b = x.shape[0]
                n += b
                val_loss_sum += float(loss.item()) * b
                for k, v in metrics.items():
                    metric_sum[k] += float(v) * b

            val_loss = val_loss_sum / max(1, n)
            val_metrics = {k: (v / max(1, n)) for k, v in metric_sum.items()}

        msg = (
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
            f"val_full={val_metrics['acc_full']*100:.2f}% "
            f"val_mean_digit={val_metrics['acc_mean_digit']*100:.2f}% "
            f"d1={val_metrics['acc_digit1']*100:.2f}% "
            f"d2={val_metrics['acc_digit2']*100:.2f}% "
            f"d3={val_metrics['acc_digit3']*100:.2f}% "
            f"d4={val_metrics['acc_digit4']*100:.2f}%"
        )
        print(msg)

        # On small datasets, full-string accuracy can stay at 0% for a long time.
        # Save best by mean per-digit accuracy, using full-string accuracy as a tie-break.
        improved = False
        if val_metrics["acc_mean_digit"] > best_val_mean_digit:
            improved = True
        elif val_metrics["acc_mean_digit"] == best_val_mean_digit and val_metrics["acc_full"] > best_val_full:
            improved = True

        if improved:
            best_val_mean_digit = val_metrics["acc_mean_digit"]
            best_val_full = val_metrics["acc_full"]
            ckpt = {
                "model_state_dict": model.state_dict(),
                "config": asdict(config),
                "best_val_acc_full": best_val_full,
                "best_val_acc_mean_digit": best_val_mean_digit,
            }
            torch.save(ckpt, out_path)
            print(
                f"Saved best checkpoint to: {out_path} (mean_digit={best_val_mean_digit*100:.2f}%, full={best_val_full*100:.2f}%)"
            )


if __name__ == "__main__":
    main()
