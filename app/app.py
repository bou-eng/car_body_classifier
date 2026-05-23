from __future__ import annotations

import time
from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
MODEL_PATH = REPO_ROOT / "models" / "smoke_mobilenetv3.pth"

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


@st.cache_resource
def load_model():
	device = get_device()
	checkpoint = torch.load(MODEL_PATH, map_location=device)

	classes = checkpoint["classes"]
	model = build_model(len(classes))
	model.load_state_dict(checkpoint["model_state_dict"])
	model.to(device)
	model.eval()

	return model, classes, device


def predict(image: Image.Image, model, classes, device):
	image_tensor = get_transform()(image).unsqueeze(0).to(device)

	start_time = time.perf_counter()
	with torch.no_grad():
		outputs = model(image_tensor)
		probabilities = torch.softmax(outputs, dim=1)[0]
	elapsed_time = time.perf_counter() - start_time

	pred_idx = int(torch.argmax(probabilities).item())
	pred_class = classes[pred_idx]
	confidence = float(probabilities[pred_idx].item())

	prob_dict = {
		classes[index]: float(probabilities[index].detach().cpu())
		for index in range(len(classes))
	}

	return pred_class, confidence, prob_dict, elapsed_time


st.set_page_config(
	page_title="Car Body Type Classifier",
	layout="wide",
)

st.title("Araba Gövde Tipi Sınıflandırma")
st.write("Bir görsel yükleyin, ardından tahmini, güven skorunu ve sınıf olasılıklarını görün.")

if not MODEL_PATH.exists():
	st.error(f"Model bulunamadı: {MODEL_PATH}")
	st.stop()

model, classes, device = load_model()
st.write(f"Kullanılan cihaz: `{device}`")

uploaded_file = st.file_uploader(
	"Bir araba görseli yükle",
	type=["jpg", "jpeg", "png", "webp"],
)

if uploaded_file is not None:
	image = Image.open(uploaded_file).convert("RGB")

	col_image, col_result = st.columns(2)

	with col_image:
		st.subheader("Yüklenen Görsel")
		st.image(image, use_container_width=True)

	with col_result:
		st.subheader("Tahmin Sonucu")

		if st.button("Tahmin Yap"):
			pred_class, confidence, prob_dict, elapsed_time = predict(
				image,
				model,
				classes,
				device,
			)

			st.success(f"Tahmin sınıfı: {pred_class}")
			st.write(f"Güven skoru: %{confidence * 100:.2f}")
			st.write(f"Tahmin süresi: {elapsed_time:.4f} saniye")

			st.subheader("Sınıf Olasılıkları")
			st.bar_chart(prob_dict)
else:
	st.info("Başlamak için bir görsel yükleyin.")
