
import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image
from model.download_model import ensure_model

st.set_page_config(page_title="KaryoAssist", page_icon="🧬", layout="centered")

st.title("🧬 KaryoAssist")
st.markdown("Upload a chromosome image to predict its class using a fine-tuned ResNet50 model.")

@st.cache_resource
def load_model():
    model_path = ensure_model()
    ckpt = torch.load(model_path, map_location="cpu")
    class_names = ckpt.get("class_names", [str(i) for i in range(1, 25)])
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    return model, class_names

model, class_names = load_model()

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]),
])

uploaded = st.file_uploader("Upload chromosome image (PNG/JPG)", type=["png","jpg","jpeg"])
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
            "Confidence": [round(float(p),4) for p in topk.values.tolist()]
        })
