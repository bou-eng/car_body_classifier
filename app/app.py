from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from PIL import Image


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from predict import (  # noqa: E402
    CLASS_TO_NUMBER,
    MODEL_PATH,
    PROJECT_CLASS_ORDER,
    get_transform,
    load_model,
)


st.set_page_config(
    page_title="Araba Gövde Tipi Sınıflandırma",
    page_icon="🚗",
    layout="wide",
)


@st.cache_resource
def load_cached_model():
    return load_model(MODEL_PATH)


def predict_uploaded_image(image: Image.Image):
    model, classes, device, checkpoint = load_cached_model()
    transform = get_transform()

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
        "model_name": checkpoint["model_name"],
        "predicted_class": pred_class,
        "predicted_number": pred_number,
        "confidence": confidence,
        "elapsed_time": elapsed_time,
        "probabilities": ordered_probabilities,
        "device": str(device),
    }


st.title("🚗 Araba Gövde Tipi Sınıflandırma")
st.write(
    "Bu arayüz, yüklenen araba görselinin gövde tipini eğitilmiş final model ile tahmin eder."
)

if not MODEL_PATH.exists():
    st.error(f"Model dosyası bulunamadı: {MODEL_PATH}")
    st.stop()

model, classes, device, checkpoint = load_cached_model()

with st.sidebar:
    st.header("Model Bilgisi")
    st.write(f"**Model:** {checkpoint.get('model_name', 'Bilinmiyor')}")
    st.write(f"**Cihaz:** {device}")
    st.write(f"**Girdi boyutu:** {checkpoint.get('image_size', 224)}x{checkpoint.get('image_size', 224)}")
    st.write(f"**Sınıf sayısı:** {len(classes)}")

    if "best_val_f1" in checkpoint:
        st.write(f"**Best Val F1:** {checkpoint['best_val_f1']:.4f}")

    st.divider()

    st.header("Sınıf Numaraları")
    for class_name in PROJECT_CLASS_ORDER:
        st.write(f"{CLASS_TO_NUMBER[class_name]} → {class_name}")


uploaded_file = st.file_uploader(
    "Bir araba görseli yükle",
    type=["jpg", "jpeg", "png", "webp", "bmp"],
)

if uploaded_file is None:
    st.info("Tahmin yapmak için bir görsel yükle.")
    st.stop()

image = Image.open(uploaded_file).convert("RGB")

left_col, right_col = st.columns([1, 1])

with left_col:
    st.subheader("Yüklenen Görsel")
    st.image(image, use_container_width=True)

with right_col:
    st.subheader("Tahmin Sonucu")

    if st.button("Tahmin Yap", type="primary"):
        result = predict_uploaded_image(image)

        st.success(f"Tahmin: {result['predicted_class']}")

        metric_col1, metric_col2, metric_col3 = st.columns(3)

        with metric_col1:
            st.metric("Sınıf Numarası", result["predicted_number"])

        with metric_col2:
            st.metric("Güven Skoru", f"%{result['confidence'] * 100:.2f}")

        with metric_col3:
            st.metric("Tahmin Süresi", f"{result['elapsed_time']:.4f} sn")

        st.write(f"**Kullanılan model:** {result['model_name']}")
        st.write(f"**Çalışma cihazı:** {result['device']}")

        st.subheader("Tüm Sınıfların Olasılık Dağılımı")

        probabilities_df = pd.DataFrame(
            {
                "Sınıf": list(result["probabilities"].keys()),
                "Olasılık": [value * 100 for value in result["probabilities"].values()],
            }
        )

        st.bar_chart(
            probabilities_df,
            x="Sınıf",
            y="Olasılık",
            use_container_width=True,
        )

        st.dataframe(
            probabilities_df.sort_values("Olasılık", ascending=False),
            use_container_width=True,
            hide_index=True,
        )