from __future__ import annotations

import argparse
import base64
import io
from dataclasses import asdict
from typing import Any, Dict

import torch
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image

from torch_captcha_data import ImagePreprocess, pil_to_tensor_rgb
from torch_captcha_model import CaptchaCNN, CaptchaConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PyTorch captcha inference service (Base64 -> text)")
    p.add_argument("--ckpt", default="./data/model/imitate_4_model_torch.pt")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8888)
    p.add_argument(
        "--max-bytes",
        type=int,
        default=1_000_000,
        help="Max request body size (bytes)",
    )
    return p.parse_args()


def load_model(ckpt_path: str) -> tuple[CaptchaCNN, CaptchaConfig, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)
    config = CaptchaConfig(**ckpt.get("config", {}))
    model = CaptchaCNN(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, config, device


def decode_base64_image(data_url_or_b64: str) -> Image.Image:
    image_data = data_url_or_b64
    if "," in image_data:
        image_data = image_data.split(",", 1)[1]
    image_bytes = base64.b64decode(image_data)
    return Image.open(io.BytesIO(image_bytes))


def create_app(model: CaptchaCNN, config: CaptchaConfig, device: torch.device, max_bytes: int) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = max_bytes
    CORS(app)

    preprocess = ImagePreprocess(image_height=config.image_height, image_width=config.image_width)

    @app.post("/predict")
    def predict() -> Any:
        data: Dict[str, Any] = request.get_json(silent=True) or {}
        image_b64 = data.get("image")
        if not image_b64:
            return jsonify({"error": "No image provided"}), 400

        try:
            img = decode_base64_image(str(image_b64))
        except Exception:
            return jsonify({"error": "Invalid base64 image"}), 400

        try:
            x = pil_to_tensor_rgb(img, preprocess).unsqueeze(0).to(device)
            with torch.inference_mode():
                pred = model.predict_text(x)[0]
            return jsonify(
                {
                    "prediction": pred,
                    "config": asdict(config),
                }
            )
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return app


def main() -> None:
    args = parse_args()
    model, config, device = load_model(args.ckpt)
    print(f"Using device: {device}")
    app = create_app(model, config, device=device, max_bytes=args.max_bytes)
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
