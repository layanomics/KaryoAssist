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
st.markdown("Upload a chromosome image to predict its class using a fine-tuned ResNet50 model.")

# ----------------------------------------------------------
# Load Model (robust handling for any save format)
# ----------------------------------------------------------
@st.cache_resource
def load_model():
    model_path = "app/models/best_resnet50_finetuned.ckpt"

    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found at: {model_path}")
        st.stop()

    ckpt = torch.load(model_path, map_location="cpu")

    # Detect the checkpoint type
    if isinstance(ckpt, torch.nn.Module):
        # Case 1: you saved the full model
        st.info("✅ Loaded a full PyTorch model object.")
        model = ckpt
        class_names = [str(i) for i in range(1, 25)]

    elif isinstance(ckpt, dict):
        # Case 2: a checkpoint dictionary
        if "state_dict" in ckpt:
            st.info("✅ Loaded model from 'state_dict' checkpoint.")
            model = models.resnet50(weights=None)
            class_names = ckpt.get("class_names", [str(i) for i in range(1, 25)])
            model.fc = nn.Linear(model.fc.in_features, len(class_names))
            model.load_state_dict(ckpt["state_dict"], strict=False)
        elif "model" in ckpt and isinstance(ckpt["model"], torch.nn.Module):
            st.info("✅ Loaded model from 'model' key in checkpoint.")
            model = ckpt["model"]
            class_names = ckpt.get("class_names", [str(i) for i in range(1, 25)])
        else:
            st.warning("⚠️ Checkpoint dict has unexpected keys — trying to reconstruct a ResNet model.")
            model = models.resnet50(weights=None)
            class_names = [str(i) for i in range(1, 25)]
            try:
                model.load_state_dict(ckpt, strict=False)
            except Exception as e:
                st.error(f"❌ Failed to load state dict automatically: {e}")
                st.stop()

    else:
        st.error("❌ Unrecognized model format.")
        st.stop()

    model.eval()
    st.success(f"✅ Model loaded successfully from: {model_path}")
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

uploaded = st.file_uploader("📤 Upload a chromosome image (PNG/JPG)", type=["png", "jpg", "jpeg"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    with st.spinner("🔬 Predicting..."):
        x = transform(img).unsqueeze(0)
        with torch.inference_mode():
            logits = model(x)
            probs = torch.softmax(logits, dim=1).squeeze(0)

        conf, idx = torch.max(probs, dim=0)
        st.success(f"**Predicted class:** {class_names[idx.item()]} | Confidence: {conf:.4f}")

        # Top-5 predictions table
        topk = torch.topk(probs, k=min(5, len(class_names)))
        st.subheader("Top-5 Predictions")
        st.table({
            "Class": [class_names[i] for i in topk.indices.tolist()],
            "Confidence": [round(float(p), 4) for p in topk.values.tolist()]
        })
else:
    st.info("⬆️ Please upload a chromosome image to begin.")

