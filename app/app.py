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
import matplotlib.pyplot as plt
import numpy as np


# ----------------------------------------------------------
# Streamlit Config
# ----------------------------------------------------------
st.set_page_config(page_title="KaryoAssist", page_icon="🧬", layout="wide")

st.sidebar.title("⚙️ Settings & Info")
st.sidebar.markdown("""
**KaryoAssist** — Automated chromosome classification + domain detection  
Model: Fine-tuned **ResNet50** on **BioImLAB**.
""")


# ----------------------------------------------------------
# FIXED DOMAIN THRESHOLD (derived from dataset percentiles)
# ----------------------------------------------------------
DOMAIN_THRESHOLD = 0.25   # images scoring <0.25 → OOD


# ----------------------------------------------------------
# Load Model
# ----------------------------------------------------------
@st.cache_resource
def load_model():
    model_path = "app/models/best_resnet50_finetuned.ckpt"

    if not os.path.exists(model_path):
        st.error("❌ Model file not found!")
        st.stop()

    ckpt = torch.load(model_path, map_location="cpu")

    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 24)

    if "model" in ckpt:
        state_dict = ckpt["model"]
    elif "state_dict" in ckpt:
        state_dict = {k.replace("model.", ""): v for k, v in ckpt["state_dict"].items()}
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict, strict=False)
    model.eval()

    class_names = [str(i) for i in range(1, 23)] + ["X", "Y"]

    st.sidebar.success("Model loaded.")
    return model, class_names


model, class_names = load_model()


# ----------------------------------------------------------
# Domain Scoring (corrected using REAL dataset stats)
# ----------------------------------------------------------
def compute_domain_score(img):
    """
    Correct domain score based on actual BioImLAB stats:

    μ_mean = 17, σ_mean = 33, dark_ratio = 0.986
    """

    gray = np.array(img.convert("L"), dtype=np.float32)
    mu = gray.mean()
    sigma = gray.std()
    dark_ratio = np.mean(gray < 150)

    # Gaussian similarity functions
    def bell(x, c, w):
        return np.exp(-((x - c) / w) ** 2)

    # centers = dataset means, widths = dataset stds
    s_mu   = bell(mu,        c=17.4,  w=11.0)
    s_sig  = bell(sigma,     c=33.1,  w=14.6)
    s_dark = bell(dark_ratio, c=0.986, w=0.03)

    score = (s_mu * s_sig * s_dark) ** (1/3)
    return float(score), float(mu), float(sigma), float(dark_ratio)


# Image quality check
def check_image_quality(img):
    w, h = img.size
    if w < 20 or h < 20:
        return "Image too small."
    return None


# Preprocessing
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ----------------------------------------------------------
# UI Tabs
# ----------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["🔬 Predict", "📈 Analytics", "ℹ️ About"])


# ----------------------------------------------------------
# Prediction Tab
# ----------------------------------------------------------
with tab1:

    st.title("🧬 KaryoAssist – Predictions")

    uploaded = st.file_uploader(
        "Upload chromosome images or a .zip folder",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff", "zip"],
        accept_multiple_files=True
    )

    if uploaded:
        results = []
        all_files = []
        tmp = tempfile.mkdtemp()

        for item in uploaded:
            if item.name.lower().endswith(".zip"):
                with zipfile.ZipFile(io.BytesIO(item.read()), "r") as z:
                    z.extractall(tmp)
                for r, _, files in os.walk(tmp):
                    for f in files:
                        if f.lower().endswith(("png", "jpg", "jpeg", "bmp", "tif", "tiff")):
                            all_files.append(os.path.join(r, f))
            else:
                all_files.append(item)

        st.info(f"Processing {len(all_files)} image(s)...")

        in_domain_count = 0
        out_domain_count = 0

        for item in all_files:

            if isinstance(item, str):
                img = Image.open(item).convert("RGB")
                name = os.path.basename(item)
            else:
                img = Image.open(item).convert("RGB")
                name = item.name

            qwarn = check_image_quality(img)
            domain_score, mu, sigma, dark = compute_domain_score(img)

            x = transform(img).unsqueeze(0)
            with torch.inference_mode():
                logits = model(x)
                probs = torch.softmax(logits, 1)[0]

            conf, idx = probs.max(0)
            conf = float(conf)
            pred = class_names[idx.item()]

            is_in_domain = (domain_score >= DOMAIN_THRESHOLD)

            # counters
            if is_in_domain and not qwarn:
                in_domain_count += 1
            else:
                out_domain_count += 1

            results.append({
                "Image": name,
                "Predicted Class": pred,
                "Confidence": round(conf, 4),
                "Domain Score": round(domain_score, 3),
                "Warning": qwarn if qwarn else ""
            })

            st.caption(f"{name}: μ={mu:.1f}, σ={sigma:.1f}, dark={dark:.3f}, score={domain_score:.3f}")

        # Show table
        df = pd.DataFrame(results)
        df["Domain Flag"] = df["Domain Score"].apply(
            lambda s: "⚠️ Out-of-domain" if s < DOMAIN_THRESHOLD else "✅ In-domain"
        )

        st.subheader("📊 Prediction Results")
        st.dataframe(df, use_container_width=True)

        # Counters
        st.markdown("### 🧠 Domain Check Summary")
        col1, col2 = st.columns(2)
        col1.metric("Likely In-domain", in_domain_count)
        col2.metric("Possibly Out-of-domain", out_domain_count)


# ----------------------------------------------------------
# Analytics Tab
# ----------------------------------------------------------
with tab2:
    st.header("📈 Analytics")
    st.write("Upload images to view analytics.")


# ----------------------------------------------------------
# About Tab
# ----------------------------------------------------------
with tab3:
    st.header("ℹ️ About")
    st.markdown("""
    **Model:** ResNet50  
    **Dataset:** BioImLAB  
    **OOD Detection:** Based on real dataset statistics extracted from 5474 images.  
    """)

