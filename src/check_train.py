from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from torchvision import models

from dataset import get_datasets


BATCH_SIZE = 16
EPOCHS = 2
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0
MODEL_PATH = Path("models/smoke_mobilenetv3.pth")


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(num_classes: int) -> nn.Module:
    try:
        weights = models.MobileNet_V3_Large_Weights.DEFAULT
        model = models.mobilenet_v3_large(weights=weights)
    except Exception:
        model = models.mobilenet_v3_large(weights=None)

    for param in model.features.parameters():
        param.requires_grad = False

    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model


def build_loaders(batch_size: int, num_workers: int, device: torch.device):
    train_dataset, val_dataset = get_datasets()
    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, train_dataset.classes, train_dataset.class_to_idx


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.detach().cpu().tolist())
        all_labels.extend(labels.detach().cpu().tolist())

    avg_loss = total_loss / max(len(loader), 1)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return avg_loss, acc, f1


def validate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.detach().cpu().tolist())
            all_labels.extend(labels.detach().cpu().tolist())

    avg_loss = total_loss / max(len(loader), 1)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return avg_loss, acc, f1


def main() -> None:
    device = get_device()
    print("Using device:", device)

    train_loader, val_loader, classes, class_to_idx = build_loaders(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        device=device,
    )

    num_classes = len(classes)
    print("Classes:", classes)
    print("Class to idx:", class_to_idx)
    print("Train samples:", len(train_loader.dataset))
    print("Val samples:", len(val_loader.dataset))

    model = build_model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    MODEL_PATH.parent.mkdir(exist_ok=True)
    best_val_f1 = -1.0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc, train_f1 = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        val_loss, val_acc, val_f1 = validate(
            model,
            val_loader,
            criterion,
            device,
        )

        print(f"\nEpoch {epoch}/{EPOCHS}")
        print(
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Train F1: {train_f1:.4f}"
        )
        print(
            f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f} | "
            f"Val   F1: {val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            checkpoint = {
                "model_name": "mobilenet_v3_large",
                "model_state_dict": model.state_dict(),
                "classes": classes,
                "class_to_idx": class_to_idx,
                "image_size": 224,
            }
            torch.save(checkpoint, MODEL_PATH)
            print("Best smoke model saved.")

    size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
    print(f"\nSaved model size: {size_mb:.2f} MB")
    print("Smoke training test OK")


if __name__ == "__main__":
    main()