from pathlib import Path
import shutil
import random

SOURCE_ROOT = Path("data/external/images_cv_car")
OUTPUT_ROOT = Path("data/candidates")

STATION_WAGON_FOLDERS = [
    "Beach wagon car",
    "Caldina car",
    "Wingroad car",
    "Libero car",
    "Legacy car",
    "Fielder car",
    "Ad wagon car",
    "Airwave car",
    "Fit shuttle car",
    "Nubira car",
]

MICRO_FOLDERS = [
    "Micro car",
    "City car",
    "Nano car",
    "Kancil car",
    "Kelisa car",
    "Cuore car",
    "Picanto car",
    "Wigo car",
    "Axia car",
    "Alto car",
    "Celerio car",
    "Mira",
    "Dayz car",
    "Moco car",
    "N box car",
    "Move car",
    "Tanto car",
    "Wagon r car",
    "Wagonr car",
    "Wagon r stingray car",
    "Wagon r fz car",
    "I miev car",
    "Qq car",
    "A star car",
    "Kwid car",
]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def collect(class_name, folder_names, limit):
    out_dir = OUTPUT_ROOT / class_name
    out_dir.mkdir(parents=True, exist_ok=True)

    files = []

    for folder_name in folder_names:
        # searches train/val/test and any nested folders
        matches = [p for p in SOURCE_ROOT.rglob("*") if p.is_dir() and p.name.lower() == folder_name.lower()]

        for folder in matches:
            for img in folder.rglob("*"):
                if img.suffix.lower() in IMAGE_EXTS:
                    files.append(img)

    random.shuffle(files)
    files = files[:limit]

    print(f"{class_name}: found {len(files)} files")

    for i, src in enumerate(files, start=1):
        ext = src.suffix.lower()
        dst = out_dir / f"{class_name}_{i:04d}{ext}"
        shutil.copy2(src, dst)

    print(f"Copied to: {out_dir}")


collect("STATION_WAGON", STATION_WAGON_FOLDERS, limit=500)
collect("MICRO", MICRO_FOLDERS, limit=900)