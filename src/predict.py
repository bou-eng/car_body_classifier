from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


MODEL_PATH = Path("models/best_model.pth")
IMAGE_SIZE = 224

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Proje PDF'indeki sabit sınıf numaraları.
CLASS_TO_NUMBER = {
    "SUV": 1,
    "VAN": 2,
    "STATION_WAGON": 3,
    "MICRO": 4,
    "OPEN_WHEEL": 5,
    "SEDAN": 6,
    "HATCHBACK": 7,
    "PICK_UP": 8,
}

PROJECT_CLASS_ORDER = [
    "SUV",
    "VAN",
    "STATION_WAGON",
    "MICRO",
    "OPEN_WHEEL",
    "SEDAN",
    "HATCHBACK",
    "PICK_UP",
]


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def build_mobilenetv3_large(num_classes: int, dropout: float = 0.4) -> nn.Module:
    model = models.mobilenet_v3_large(weights=None)

    model.classifier[2] = nn.Dropout(p=dropout, inplace=True)

    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)

    return model


def build_efficientnet_b0(num_classes: int, dropout: float = 0.4) -> nn.Module:
    model = models.efficientnet_b0(weights=None)

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(in_features, num_classes),
    )

    return model


def build_model(model_name: str, num_classes: int, dropout: float = 0.4) -> nn.Module:
    if model_name == "mobilenetv3_large":
        return build_mobilenetv3_large(num_classes=num_classes, dropout=dropout)

    if model_name in {"efficientnet_b0", "efficientnet_b0_v2"}:
        return build_efficientnet_b0(num_classes=num_classes, dropout=dropout)

    raise ValueError(f"Desteklenmeyen model adı: {model_name}")


def load_model(model_path: Path = MODEL_PATH):
    device = get_device()

    if not model_path.exists():
        raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)

    model_name = checkpoint["model_name"]
    classes = checkpoint["classes"]
    num_classes = len(classes)
    dropout = checkpoint.get("dropout", 0.4)

    model = build_model(
        model_name=model_name,
        num_classes=num_classes,
        dropout=dropout,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, classes, device, checkpoint


def get_transform():
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def predict_image(image_path: str | Path, model_path: str | Path = MODEL_PATH):
    image_path = Path(image_path)
    model_path = Path(model_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Görsel bulunamadı: {image_path}")

    model, classes, device, checkpoint = load_model(model_path)

    transform = get_transform()

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    start_time = time.time()

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

    elapsed_time = time.time() - start_time

    pred_idx = int(torch.argmax(probabilities).item())
    pred_class = classes[pred_idx]
    pred_number = CLASS_TO_NUMBER[pred_class]
    confidence = float(probabilities[pred_idx].detach().cpu().item())

    probabilities_by_class = {
        classes[i]: float(probabilities[i].detach().cpu().item())
        for i in range(len(classes))
    }

    ordered_probabilities = {
        class_name: probabilities_by_class.get(class_name, 0.0)
        for class_name in PROJECT_CLASS_ORDER
    }

    return {
        "image_path": str(image_path),
        "model_name": checkpoint["model_name"],
        "predicted_class": pred_class,
        "predicted_number": pred_number,
        "confidence": confidence,
        "probabilities": ordered_probabilities,
        "elapsed_time": elapsed_time,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Tek bir araba görseli için gövde tipi tahmini yapar."
    )

    parser.add_argument(
        "image_path",
        type=str,
        help="Tahmin yapılacak görselin yolu.",
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default=str(MODEL_PATH),
        help="Model checkpoint yolu. Varsayılan: models/best_model.pth",
    )

    args = parser.parse_args()

    result = predict_image(
        image_path=args.image_path,
        model_path=args.model_path,
    )

    print("\n========== TAHMİN SONUCU ==========")
    print(f"Görsel: {result['image_path']}")
    print(f"Model: {result['model_name']}")
    print(f"Tahmin edilen sınıf: {result['predicted_class']}")
    print(f"Sınıf numarası: {result['predicted_number']}")
    print(f"Güven skoru: %{result['confidence'] * 100:.2f}")
    print(f"Tahmin süresi: {result['elapsed_time']:.4f} saniye")

    print("\nSınıf olasılıkları:")
    for class_name, probability in result["probabilities"].items():
        print(f"{class_name}: %{probability * 100:.2f}")


if __name__ == "__main__":
    main()