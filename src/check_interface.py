from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "smoke_mobilenetv3.pth"
DEFAULT_SAMPLE_DIR = REPO_ROOT / "data" / "processed" / "val"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(num_classes: int) -> nn.Module:
    model = models.mobilenet_v3_large(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model


def get_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def load_model(model_path: Path, device: torch.device):
    checkpoint = torch.load(model_path, map_location=device)

    classes = checkpoint["classes"]
    model = build_model(num_classes=len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, classes


def find_sample_image(sample_dir: Path) -> Path | None:
    for extension in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        matches = sorted(sample_dir.rglob(extension))
        if matches:
            return matches[0]
    return None


def predict(image: Image.Image, model, classes, device):
    transform = get_transform()
    image_tensor = transform(image).unsqueeze(0).to(device)

    start_time = time.perf_counter()
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
    elapsed_time = time.perf_counter() - start_time

    pred_idx = int(torch.argmax(probabilities).item())
    pred_class = classes[pred_idx]
    confidence = float(probabilities[pred_idx].item())

    return pred_class, confidence, probabilities.detach().cpu(), elapsed_time


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test for the car body classification interface")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to the trained checkpoint",
    )
    parser.add_argument(
        "--image-path",
        type=Path,
        default=None,
        help="Path to a sample image to run through the interface pipeline",
    )
    args = parser.parse_args()

    if not args.model_path.exists():
        print(f"Model not found: {args.model_path}")
        return 1

    image_path = args.image_path
    if image_path is None:
        image_path = find_sample_image(DEFAULT_SAMPLE_DIR)

    if image_path is None or not image_path.exists():
        print(f"Sample image not found under: {DEFAULT_SAMPLE_DIR}")
        return 1

    device = get_device()
    model, classes = load_model(args.model_path, device)

    image = Image.open(image_path).convert("RGB")
    pred_class, confidence, probabilities, elapsed_time = predict(image, model, classes, device)

    top_k = min(3, len(classes))
    top_probs = torch.topk(probabilities, k=top_k)

    print("Interface smoke test passed")
    print(f"Device: {device}")
    print(f"Model: {args.model_path}")
    print(f"Image: {image_path}")
    print(f"Predicted class: {pred_class}")
    print(f"Confidence: {confidence * 100:.2f}%")
    print(f"Inference time: {elapsed_time:.4f} seconds")
    print("Top predictions:")
    for score, index in zip(top_probs.values.tolist(), top_probs.indices.tolist()):
        print(f"  {classes[index]}: {score * 100:.2f}%")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())