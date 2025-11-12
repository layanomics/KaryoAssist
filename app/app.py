import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image
import os
import pandas as pd
import zipfile
import io
import tempfile

# ----------------------------------------------------------
# Streamlit Page Config
# ----------------------------------------------------------
st.set_page_config(page_title="KaryoAssist", page_icon="🧬", layout="wide")
st.title("🧬 KaryoAssist")
st.markdown("Upload **images or an entire folder (.zip)** of chromosome samples to predict their classes using your fine-tuned **ResNet50** model (trained on BioImLAB dataset).")

# ----------------------------------------------------------
# Load Model (24 chromosome classes)
# ----------------------------------------------------------
@st.cache_resource
def load_model():
    model_path = "app/models/best_resnet50_finetuned.ckpt"

    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found at: {model_path}")
        st.stop()

    ckpt = torch.load(model_path, map_location="cpu")

    # Build architecture
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 24)

    # Load checkpoint safely
    if "model" in ckpt:
        state_dict = ckpt["model"]
    elif "state_dict" in ckpt:
        state_dict = {k.replace("model.", ""): v for k, v in ckpt["state_dict"].items()}
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # Real chromosome labels
    class_names = [str(i) for i in range(1, 23)] + ["X", "Y"]

    st.success(f"✅ Model loaded successfully from: {model_path}")
    st.caption(f"Detected {len(class_names)} chromosome classes: 1–22, X, Y")
    return model, class_names

# ----------------------------------------------------------
# Initialize model once
# ----------------------------------------------------------
model, class_names = load_model()

# ----------------------------------------------------------
# Image Preprocessing
# ----------------------------------------------------------
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]),
])

# ----------------------------------------------------------
# File or Folder Upload
# ----------------------------------------------------------
uploaded_items = st.file_uploader(
    "📁 Upload images (PNG/JPG/BMP) or a folder as .zip",
    type=["png", "jpg", "jpeg", "bmp", "zip"],
    accept_multiple_files=True
)

# ----------------------------------------------------------
# Batch Prediction
# ----------------------------------------------------------
if uploaded_items:
    results = []
    image_files = []

    # Temporary extraction directory for ZIPs
    temp_dir = tempfile.mkdtemp()

    # Gather all images (individual or inside ZIP)
    for item in uploaded_items:
        if item.name.lower().endswith(".zip"):
            st.info(f"📦 Extracting ZIP folder: {item.name}")
            with zipfile.ZipFile(io.BytesIO(item.read()), "r") as zf:
                zf.extractall(temp_dir)
            for root, _, files in os.walk(temp_dir):
                for f in files:
                    if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                        image_files.append(os.path.join(root, f))
        else:
            image_files.append(item)

    if not image_files:
        st.warning("⚠️ No valid image files found. Please upload PNG, JPG, or BMP images.")
        st.stop()

    st.info(f"🔬 Processing {len(image_files)} image(s)...")

    for img_item in image_files:
        try:
            # Handle files from zip or direct upload
            if isinstance(img_item, str):
                img = Image.open(img_item).convert("RGB")
                name = os.path.basename(img_item)
            else:
                img = Image.open(img_item).convert("RGB")
                name = img_item.name

            x = transform(img).unsqueeze(0)
            with torch.inference_mode():
                logits = model(x)
                probs = torch.softmax(logits, dim=1).squeeze(0)

            conf, idx = torch.max(probs, dim=0)
            pred_label = class_names[idx.item()]

            results.append({
                "Image": name,
                "Predicted Class": pred_label,
                "Confidence": round(float(conf), 4),
                "Preview": img
            })

        except Exception as e:
            st.error(f"❌ Error processing {name}: {e}")

    # ------------------------------------------------------
    # Display results table
    # ------------------------------------------------------
    if results:
        df = pd.DataFrame([{
            "Image": r["Image"],
            "Predicted Class": r["Predicted Class"],
            "Confidence": r["Confidence"]
        } for r in results])

        st.subheader("📊 Prediction Results")
        st.dataframe(df, use_container_width=True)

        # ------------------------------------------------------
        # Show image previews in a grid
        # ------------------------------------------------------
        st.subheader("🖼️ Image Previews")
        cols = st.columns(3)
        for i, r in enumerate(results):
            with cols[i % 3]:
                st.image(r["Preview"], caption=f"{r['Image']} → {r['Predicted Class']} ({r['Confidence']:.3f})", use_column_width=True)

else:
    st.info("⬆️ Please upload individual chromosome images or a folder (.zip) to begin.")
