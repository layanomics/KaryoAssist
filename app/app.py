import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image
import os

st.set_page_config(page_title="KaryoAssist", page_icon="🧬", layout="centered")

st.title("🧬 KaryoAssist")
st.markdown("Upload a chromosome image to predict its class using a fine-tuned ResNet50 model.")

# ----------------------------------------------------------
# Load model directly from repo (no download script needed)
# ----------------------------------------------------------
@st.cache_resource
def load_model():
    model_path = "app/models/best_resnet50_finetuned.ckpt"

    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found at: {model_path}")
        st.stop()

    ckpt = torch.load(model_path, map_location="cpu")

    # handle both full model and state_dict saves
    if not isinstance(ckpt, dict):
        model = ckpt
        class_names = [str(i) for i in range(1, 25)]
    else:
        model = models.resnet50(weights=None)
        class_names = ckpt.get("class_names", [str(i) for i in range(1, 25)])
        model.fc = nn.Linear(model.fc.in_features, len(class_names))
        model.load_state_dict(ckpt["state_dict"], strict=False)

    model.eval()
    st.info(f"✅ Model loaded successfully from: {model_path}")
    return model, class_names

model, class_names = load_model()

# ----------------------------------------------------------
# Image preprocessing and prediction
# ----------------------------------------------------------
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]),
])

uploaded = st.file_uploader("Upload chromosome image (PNG/JPG)", type=["png", "jpg", "jpeg"])
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Predicting..."):
        x = transform(img).unsqueeze(0)
        with torch.inference_mode():
            logits = model(x)
            probs = torch.softmax(logits, dim=1).squeeze(0)

        conf, idx = torch.max(probs, dim=0)
        st.success(f"**Predicted class:** {class_names[idx.item()]} | Confidence: {conf:.4f}")

        topk = torch.topk(probs, k=min(5, len(class_names)))
        st.subheader("Top-5 Predictions")
        st.table({
            "Class": [class_names[i] for i in topk.indices.tolist()],
            "Confidence": [round(float(p), 4) for p in topk.values.tolist()]
        })

