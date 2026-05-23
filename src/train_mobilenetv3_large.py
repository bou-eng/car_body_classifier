from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader
from torchvision import models

from dataset import get_datasets


# =========================
# CONFIG
# =========================
BATCH_SIZE = 32
EPOCHS = 20
FROZEN_EPOCHS = 3

CLASSIFIER_LR = 5e-4
FINE_TUNE_LR = 3e-5
WEIGHT_DECAY = 2e-4

NUM_WORKERS = 2
IMAGE_SIZE = 224
EARLY_STOPPING_PATIENCE = 5

DROPOUT = 0.4
LABEL_SMOOTHING = 0.1

DRIVE_ROOT = Path("/content/drive/MyDrive/car_body_project")

MODEL_PATH = DRIVE_ROOT / "models" / "best_mobilenetv3_large.pth"
LAST_MODEL_PATH = DRIVE_ROOT / "models" / "last_mobilenetv3_large.pth"

OUTPUT_DIR = DRIVE_ROOT / "outputs" / "mobilenetv3_large"
LOG_PATH = DRIVE_ROOT / "logs" / "mobilenetv3_large_training_log.csv"


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(num_classes: int, freeze_backbone: bool = True) -> nn.Module:
    try:
        weights = models.MobileNet_V3_Large_Weights.DEFAULT
        model = models.mobilenet_v3_large(weights=weights)
        print("Loaded MobileNetV3-Large with pretrained ImageNet weights.")
    except Exception as error:
        print("Pretrained weights could not be loaded. Using random weights.")
        print("Error:", error)
        model = models.mobilenet_v3_large(weights=None)

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    model.classifier[2] = nn.Dropout(p=DROPOUT, inplace=True)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)

    return model


def unfreeze_model(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = True


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


def build_class_weights(classes, device):
    weights = torch.ones(len(classes), dtype=torch.float32)

    for idx, class_name in enumerate(classes):
        if class_name == "SEDAN":
            weights[idx] = 1.8
        elif class_name == "HATCHBACK":
            weights[idx] = 1.2
        elif class_name == "SUV":
            weights[idx] = 1.1
        elif class_name == "STATION_WAGON":
            weights[idx] = 0.9
        else:
            weights[idx] = 1.0

    print("Class weights:")
    for class_name, weight in zip(classes, weights.tolist()):
        print(f"{class_name}: {weight}")

    return weights.to(device)


def train_one_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    use_amp = device.type == "cuda"

    for batch_idx, (images, labels) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.detach().cpu().tolist())
        all_labels.extend(labels.detach().cpu().tolist())

        if batch_idx % 20 == 0:
            print(f"Batch {batch_idx}/{len(loader)} | Loss: {loss.item():.4f}")

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
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.detach().cpu().tolist())
            all_labels.extend(labels.detach().cpu().tolist())

    avg_loss = total_loss / max(len(loader), 1)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return avg_loss, acc, f1, all_labels, all_preds


def save_training_log(history):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(LOG_PATH, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "epoch",
                "train_loss",
                "val_loss",
                "train_acc",
                "val_acc",
                "train_f1",
                "val_f1",
                "learning_rate",
            ],
        )
        writer.writeheader()
        writer.writerows(history)


def save_curves(history):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    epochs = [row["epoch"] for row in history]

    train_loss = [row["train_loss"] for row in history]
    val_loss = [row["val_loss"] for row in history]

    train_acc = [row["train_acc"] for row in history]
    val_acc = [row["val_acc"] for row in history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss - MobileNetV3-Large")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "loss_curve.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_acc, label="Train Accuracy")
    plt.plot(epochs, val_acc, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training & Validation Accuracy - MobileNetV3-Large")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "accuracy_curve.png", dpi=200)
    plt.close()


def save_confusion_matrix(y_true, y_pred, classes):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))

    cm_sum = cm.sum(axis=1, keepdims=True)
    cm_normalized = np.divide(
        cm,
        cm_sum,
        out=np.zeros_like(cm, dtype=float),
        where=cm_sum != 0,
    )

    plt.figure(figsize=(10, 8))
    plt.imshow(cm_normalized, interpolation="nearest")
    plt.title("Normalized Confusion Matrix - MobileNetV3-Large")
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha="right")
    plt.yticks(tick_marks, classes)

    for i in range(cm_normalized.shape[0]):
        for j in range(cm_normalized.shape[1]):
            plt.text(
                j,
                i,
                f"{cm_normalized[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
            )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix_normalized.png", dpi=200)
    plt.close()


def save_metrics(y_true, y_pred, classes, best_epoch, best_val_f1, model_size_mb):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(classes))),
        target_names=classes,
        zero_division=0,
        output_dict=True,
    )

    report_text = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(classes))),
        target_names=classes,
        zero_division=0,
    )

    metrics = {
        "model_name": "mobilenetv3_large",
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "weighted_recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "model_size_mb": model_size_mb,
        "dropout": DROPOUT,
        "label_smoothing": LABEL_SMOOTHING,
        "fine_tune_lr": FINE_TUNE_LR,
        "classification_report": report_dict,
    }

    with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4, ensure_ascii=False)

    with open(OUTPUT_DIR / "classification_report.txt", "w", encoding="utf-8") as file:
        file.write(report_text)


def main() -> None:
    device = get_device()
    print("Using device:", device)

    if device.type != "cuda":
        print("WARNING: CUDA is not active. Training will be slow.")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

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

    model = build_model(num_classes=num_classes, freeze_backbone=True).to(device)

    class_weights = build_class_weights(classes, device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=LABEL_SMOOTHING,
    )

    optimizer = torch.optim.AdamW(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=CLASSIFIER_LR,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    best_val_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, EPOCHS + 1):
        print(f"\n========== Epoch {epoch}/{EPOCHS} ==========")

        if epoch == FROZEN_EPOCHS + 1:
            print("Unfreezing MobileNetV3-Large backbone for fine-tuning...")
            unfreeze_model(model)

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=FINE_TUNE_LR,
                weight_decay=WEIGHT_DECAY,
            )

            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=0.5,
                patience=2,
            )

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Learning rate: {current_lr}")

        train_loss, train_acc, train_f1 = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
        )

        val_loss, val_acc, val_f1, y_true, y_pred = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        scheduler.step(val_f1)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "train_f1": train_f1,
            "val_f1": val_f1,
            "learning_rate": current_lr,
        }

        history.append(row)
        save_training_log(history)
        save_curves(history)

        print(
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Train F1: {train_f1:.4f}"
        )
        print(
            f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f} | "
            f"Val   F1: {val_f1:.4f}"
        )

        last_checkpoint = {
            "model_name": "mobilenetv3_large",
            "model_state_dict": model.state_dict(),
            "classes": classes,
            "class_to_idx": class_to_idx,
            "image_size": IMAGE_SIZE,
            "epoch": epoch,
            "val_f1": val_f1,
            "dropout": DROPOUT,
            "label_smoothing": LABEL_SMOOTHING,
        }

        torch.save(last_checkpoint, LAST_MODEL_PATH)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            epochs_without_improvement = 0

            best_checkpoint = {
                "model_name": "mobilenetv3_large",
                "model_state_dict": model.state_dict(),
                "classes": classes,
                "class_to_idx": class_to_idx,
                "image_size": IMAGE_SIZE,
                "epoch": epoch,
                "best_val_f1": best_val_f1,
                "dropout": DROPOUT,
                "label_smoothing": LABEL_SMOOTHING,
            }

            torch.save(best_checkpoint, MODEL_PATH)
            print(f"Best model saved. Best Val F1: {best_val_f1:.4f}")
        else:
            epochs_without_improvement += 1
            print(f"No improvement count: {epochs_without_improvement}/{EARLY_STOPPING_PATIENCE}")

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    print("\nLoading best model for final validation report...")

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    final_val_loss, final_val_acc, final_val_f1, final_y_true, final_y_pred = validate(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
    )

    model_size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)

    save_confusion_matrix(final_y_true, final_y_pred, classes)
    save_metrics(
        y_true=final_y_true,
        y_pred=final_y_pred,
        classes=classes,
        best_epoch=best_epoch,
        best_val_f1=best_val_f1,
        model_size_mb=model_size_mb,
    )

    print("\n========== TRAINING FINISHED ==========")
    print(f"Best epoch: {best_epoch}")
    print(f"Best Val F1: {best_val_f1:.4f}")
    print(f"Final Val Loss: {final_val_loss:.4f}")
    print(f"Final Val Acc : {final_val_acc:.4f}")
    print(f"Final Val F1  : {final_val_f1:.4f}")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Model size: {model_size_mb:.2f} MB")
    print(f"Outputs saved to: {OUTPUT_DIR}")
    print(f"Log saved to: {LOG_PATH}")


if __name__ == "__main__":
    main()
