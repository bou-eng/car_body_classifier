from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from config import (
    BATCH_SIZE,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    TRAIN_DIR,
    VAL_DIR,
)


def get_train_transform():
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                IMAGE_SIZE,
                scale=(0.75, 1.0),
                ratio=(0.85, 1.15),
                antialias=True,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(
                brightness=0.25,
                contrast=0.25,
                saturation=0.25,
                hue=0.05,
            ),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05),
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_val_transform():
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_datasets(train_dir=TRAIN_DIR, val_dir=VAL_DIR):
    train_dataset = datasets.ImageFolder(
        root=train_dir,
        transform=get_train_transform(),
    )
    val_dataset = datasets.ImageFolder(
        root=val_dir,
        transform=get_val_transform(),
    )

    return train_dataset, val_dataset


def get_dataloaders(train_dir=TRAIN_DIR, val_dir=VAL_DIR, batch_size=BATCH_SIZE):
    train_dataset, val_dataset = get_datasets(train_dir=train_dir, val_dir=val_dir)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0 ,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, val_loader, train_dataset.classes
