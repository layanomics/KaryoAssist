import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image
import os

# ----------------------------------------------------------
# Streamlit Page Config
# ----------------------------------------------------------
st.set_page_config(page_title="KaryoAssist", page_icon="🧬", layout="centered")
st.title("🧬 KaryoAssist")
st.markdown("Upload a chromosome image to predict its class using your fine-tuned **ResNet50** model (trained on BioImLAB dataset).")

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
# Image Preprocessing & Prediction
# ----------------------------------------------------------
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]),
])

uploaded = st.file_uploader("📤 Upload a chromosome image (PNG/JPG/BMP)", type=["png", "jpg", "jpeg", "bmp"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    with st.spinner("🔬 Predicting..."):
        x = transform(img).unsqueeze(0)
        with torch.inference_mode():
            logits = model(x)
            probs = torch.softmax(logits, dim=1).squeeze(0)

        conf, idx = torch.max(probs, dim=0)
        idx = idx.item()

        if idx < len(class_names):
            pred_class = class_names[idx]
            st.success(f"**Predicted class:** {pred_class} | Confidence: {conf:.4f}")
        else:
            st.warning(f"⚠️ Predicted index {idx} is out of range (model output mismatch).")

        # Top-5 predictions
        topk = torch.topk(probs, k=5)
        st.subheader("Top-5 Predictions")
        st.table({
            "Class": [class_names[i] if i < len(class_names) else f"Unknown({i})" for i in topk.indices.tolist()],
            "Confidence": [round(float(p), 4) for p in topk.values.tolist()]
        })
else:
    st.info("⬆️ Please upload a chromosome image to begin.")
