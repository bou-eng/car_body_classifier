from __future__ import annotations

import csv
import shutil
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

from dataset import get_datasets


DRIVE_ROOT = Path("/content/drive/MyDrive/car_body_project")

MODEL_PATH = DRIVE_ROOT / "models" / "best_mobilenetv3_large.pth"
OUTPUT_DIR = DRIVE_ROOT / "outputs" / "mobilenetv3_large" / "error_analysis"

MISCLASSIFIED_CSV = OUTPUT_DIR / "misclassified_predictions.csv"
SUMMARY_CSV = OUTPUT_DIR / "confusion_pair_summary.csv"

DROPOUT = 0.4
TOP_K = 3


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(num_classes: int) -> nn.Module:
    model = models.mobilenet_v3_large(weights=None)

    model.classifier[2] = nn.Dropout(p=DROPOUT, inplace=True)

    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)

    return model


def safe_copy_image(source_path: Path, target_dir: Path, new_name: str) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / new_name
    shutil.copy2(source_path, target_path)


def main() -> None:
    device = get_device()
    print("Using device:", device)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_dataset, val_dataset = get_datasets()

    checkpoint = torch.load(MODEL_PATH, map_location=device)

    classes = checkpoint["classes"]
    num_classes = len(classes)

    print("Checkpoint classes:", classes)
    print("Validation dataset classes:", val_dataset.classes)
    print("Validation samples:", len(val_dataset))

    model = build_model(num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    misclassified_rows = []
    pair_counts = {}

    with torch.no_grad():
        for index in range(len(val_dataset)):
            image_tensor, true_idx = val_dataset[index]
            image_path = Path(val_dataset.samples[index][0])

            image_tensor = image_tensor.unsqueeze(0).to(device)

            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]

            pred_idx = int(torch.argmax(probabilities).item())
            confidence = float(probabilities[pred_idx].item())

            true_class = classes[true_idx]
            pred_class = classes[pred_idx]

            top_probs, top_indices = torch.topk(probabilities, k=TOP_K)

            top_predictions = []
            for prob, cls_idx in zip(top_probs, top_indices):
                top_predictions.append(
                    f"{classes[int(cls_idx.item())]}:{float(prob.item()):.4f}"
                )

            if pred_idx != true_idx:
                pair_key = (true_class, pred_class)
                pair_counts[pair_key] = pair_counts.get(pair_key, 0) + 1

                pair_dir = OUTPUT_DIR / "images" / f"{true_class}_predicted_{pred_class}"

                new_name = f"{index:04d}_{image_path.name}"
                copied_path = pair_dir / new_name

                safe_copy_image(image_path, pair_dir, new_name)

                misclassified_rows.append(
                    {
                        "index": index,
                        "filename": image_path.name,
                        "original_path": str(image_path),
                        "true_class": true_class,
                        "predicted_class": pred_class,
                        "confidence": confidence,
                        "top3": " | ".join(top_predictions),
                        "copied_to": str(copied_path),
                    }
                )

    with open(MISCLASSIFIED_CSV, "w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "index",
            "filename",
            "original_path",
            "true_class",
            "predicted_class",
            "confidence",
            "top3",
            "copied_to",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(misclassified_rows)

    summary_rows = []

    for (true_class, pred_class), count in sorted(
        pair_counts.items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        summary_rows.append(
            {
                "true_class": true_class,
                "predicted_class": pred_class,
                "count": count,
            }
        )

    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as file:
        fieldnames = ["true_class", "predicted_class", "count"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    total_val = len(val_dataset)
    total_wrong = len(misclassified_rows)
    total_correct = total_val - total_wrong

    print("\n========== MOBILENETV3 ERROR ANALYSIS FINISHED ==========")
    print(f"Total validation images: {total_val}")
    print(f"Correct predictions: {total_correct}")
    print(f"Wrong predictions: {total_wrong}")
    print(f"Error rate: {total_wrong / total_val:.4f}")

    print(f"\nMisclassified CSV saved to:")
    print(MISCLASSIFIED_CSV)

    print(f"\nConfusion pair summary saved to:")
    print(SUMMARY_CSV)

    print(f"\nMisclassified images saved under:")
    print(OUTPUT_DIR / "images")

    print("\nTop confusion pairs:")
    for row in summary_rows[:15]:
        print(
            f"{row['true_class']} -> {row['predicted_class']}: {row['count']}"
        )


if __name__ == "__main__":
    main()
