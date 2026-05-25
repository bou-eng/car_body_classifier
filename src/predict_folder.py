from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch
from PIL import Image

from predict import (
    CLASS_TO_NUMBER,
    MODEL_PATH,
    PROJECT_CLASS_ORDER,
    get_transform,
    load_model,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def collect_images(input_dir: Path, recursive: bool = False) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Görsel klasörü bulunamadı: {input_dir}")

    if recursive:
        image_paths = [
            path for path in input_dir.rglob("*")
            if path.suffix.lower() in IMAGE_EXTENSIONS
        ]
    else:
        image_paths = [
            path for path in input_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]

    return sorted(image_paths)


def predict_folder(
    input_dir: str | Path,
    output_txt: str | Path = "outputs/preds.txt",
    details_csv: str | Path = "outputs/prediction_details.csv",
    model_path: str | Path = MODEL_PATH,
    recursive: bool = False,
) -> None:
    input_dir = Path(input_dir)
    output_txt = Path(output_txt)
    details_csv = Path(details_csv)
    model_path = Path(model_path)

    output_txt.parent.mkdir(parents=True, exist_ok=True)
    details_csv.parent.mkdir(parents=True, exist_ok=True)

    model, classes, device, checkpoint = load_model(model_path)
    transform = get_transform()

    image_paths = collect_images(input_dir=input_dir, recursive=recursive)

    if not image_paths:
        raise ValueError(f"Bu klasörde görsel bulunamadı: {input_dir}")

    print(f"Model: {checkpoint['model_name']}")
    print(f"Device: {device}")
    print(f"Image count: {len(image_paths)}")
    print(f"Output txt: {output_txt}")
    print(f"Details csv: {details_csv}")

    txt_lines = []
    csv_rows = []

    total_start_time = time.time()

    with torch.no_grad():
        for image_path in image_paths:
            image = Image.open(image_path).convert("RGB")
            image_tensor = transform(image).unsqueeze(0).to(device)

            start_time = time.time()

            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]

            elapsed_time = time.time() - start_time

            pred_idx = int(torch.argmax(probabilities).item())
            pred_class = classes[pred_idx]
            pred_number = CLASS_TO_NUMBER[pred_class]
            confidence = float(probabilities[pred_idx].detach().cpu().item())

            # Test script için kritik format:
            # filename.jpg | Tahmin: 6
            txt_lines.append(f"{image_path.name} | Tahmin: {pred_number}")

            row = {
                "filename": image_path.name,
                "path": str(image_path),
                "predicted_class": pred_class,
                "predicted_number": pred_number,
                "confidence": confidence,
                "elapsed_time": elapsed_time,
            }

            for class_name in PROJECT_CLASS_ORDER:
                if class_name in classes:
                    class_index = classes.index(class_name)
                    row[f"prob_{class_name}"] = float(
                        probabilities[class_index].detach().cpu().item()
                    )
                else:
                    row[f"prob_{class_name}"] = 0.0

            csv_rows.append(row)

            print(
                f"{image_path.name} -> {pred_class} "
                f"({pred_number}) | confidence={confidence:.4f} | "
                f"time={elapsed_time:.4f}s"
            )

    with open(output_txt, "w", encoding="utf-8") as file:
        file.write("\n".join(txt_lines))
        file.write("\n")

    fieldnames = [
        "filename",
        "path",
        "predicted_class",
        "predicted_number",
        "confidence",
        "elapsed_time",
    ] + [f"prob_{class_name}" for class_name in PROJECT_CLASS_ORDER]

    with open(details_csv, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    total_elapsed_time = time.time() - total_start_time

    print("\n========== FOLDER PREDICTION FINISHED ==========")
    print(f"Total images: {len(image_paths)}")
    print(f"Total time: {total_elapsed_time:.4f} seconds")
    print(f"Average time per image: {total_elapsed_time / len(image_paths):.4f} seconds")
    print(f"preds.txt saved to: {output_txt}")
    print(f"details csv saved to: {details_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bir klasördeki araba görselleri için preds.txt üretir."
    )

    parser.add_argument(
        "input_dir",
        type=str,
        help="Tahmin yapılacak görsellerin bulunduğu klasör.",
    )

    parser.add_argument(
        "--output-txt",
        type=str,
        default="outputs/preds.txt",
        help="Oluşturulacak preds.txt yolu.",
    )

    parser.add_argument(
        "--details-csv",
        type=str,
        default="outputs/prediction_details.csv",
        help="Detaylı tahmin CSV çıktısı.",
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default=str(MODEL_PATH),
        help="Model checkpoint yolu.",
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Alt klasörlerdeki görselleri de tahmin et.",
    )

    args = parser.parse_args()

    predict_folder(
        input_dir=args.input_dir,
        output_txt=args.output_txt,
        details_csv=args.details_csv,
        model_path=args.model_path,
        recursive=args.recursive,
    )


if __name__ == "__main__":
    main()