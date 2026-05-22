# src/config.py

# Görsel boyutu
IMAGE_SIZE = 224

# Veri klasörleri
TRAIN_DIR = "data/processed/train"
VAL_DIR = "data/processed/val"

# ImageNet üzerinde pretrained modeller için standart normalizasyon
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Eğitim ayarları
BATCH_SIZE = 32
NUM_CLASSES = 8
EPOCHS = 25
LEARNING_RATE = 1e-4

# Kullanılacak model
MODEL_NAME = "efficientnet_b0"

# Sınıf isimleri
CLASS_NAMES = [
    "SUV",
    "VAN",
    "STATION_WAGON",
    "MICRO",
    "OPEN_WHEEL",
    "SEDAN",
    "HATCHBACK",
    "PICK_UP"
]

# Test scriptinde kullanılacak sınıf numaraları
CLASS_TO_IDX = {
    "SUV": 1,
    "VAN": 2,
    "STATION_WAGON": 3,
    "MICRO": 4,
    "OPEN_WHEEL": 5,
    "SEDAN": 6,
    "HATCHBACK": 7,
    "PICK_UP": 8
}

# Numara -> sınıf adı dönüşümü
IDX_TO_CLASS = {
    1: "SUV",
    2: "VAN",
    3: "STATION_WAGON",
    4: "MICRO",
    5: "OPEN_WHEEL",
    6: "SEDAN",
    7: "HATCHBACK",
    8: "PICK_UP"
}
