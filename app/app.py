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
import numpy as np  # 👈 for quality checks

# ----------------------------------------------------------
# Streamlit Page Config
# ----------------------------------------------------------
st.set_page_config(page_title="KaryoAssist", page_icon="🧬", layout="wide")

# Sidebar Info
st.sidebar.title("⚙️ Settings & Info")
st.sidebar.markdown("""
**KaryoAssist** — an AI-powered assistant for automated chromosome classification.  
Upload single images, multiple images, or a `.zip` folder for batch prediction.
""")
st.sidebar.info("Model: Fine-tuned **ResNet50** (24 chromosome classes: 1–22, X, Y).")

# 🔧 Domain sensitivity (single source of truth)
domain_threshold = st.sidebar.slider(
    "Domain threshold (higher = stricter OOD)",
    min_value=0.00, max_value=1.00, value=0.30, step=0.01
)

# ----------------------------------------------------------
# Tabs Layout
# ----------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["🔬 Predict", "📈 Analytics", "ℹ️ About Model"])

with tab1:
    st.title("🧬 KaryoAssist")
    st.markdown(
        "Upload **images or an entire folder (.zip)** of chromosome samples to "
        "predict their classes using your fine-tuned **ResNet50** model (trained on BioImLAB dataset)."
    )

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

    st.sidebar.success("✅ Model loaded successfully")
    st.sidebar.caption(f"Detected {len(class_names)} classes: 1–22, X, Y")
    return model, class_names


# ----------------------------------------------------------
# Image Quality / Domain Check
# ----------------------------------------------------------
def check_image_quality(img, min_size=20, min_contrast=8):
    """Return warning string if image looks out-of-domain or poor quality."""
    w, h = img.size
    if w < min_size or h < min_size:
        return "Image too small (likely not a chromosome crop)."
    gray = np.array(img.convert("L"))
    contrast = gray.std()
    if contrast < min_contrast:
        return f"Low contrast (σ={contrast:.1f}) – may be out of domain."
    return None


def compute_domain_score(img):
    """
    Domain score calibrated from REAL BioImLAB stats.

    Based on your dataset profile:
    - mean_intensity (μ):  ~17.4 (5–95% ≈ 1.1–36.0)
    - std_intensity  (σ):  ~33.1 (5–95% ≈ 9.1–56.8)
    - dark_ratio          ≈ 0.99 (5% ≈ 0.923, most values ~1.0)

    Returns: (score, mu, sigma, dark_ratio), where score ∈ [0, 1]
             higher score → more in-domain.
    """
    gray = np.array(img.convert("L"), dtype=np.float32)
    mu = gray.mean()
    sigma = gray.std()
    dark_ratio = np.mean(gray < 128)

    # Gaussian-like similarity around dataset centers
    def bell(x, c, w):
        return np.exp(-0.5 * ((x - c) / w) ** 2)

    # Centers from dataset summary / percentiles
    s_mu = bell(mu, c=17.4, w=15.0)          # brightness
    s_sig = bell(sigma, c=33.1, w=20.0)      # contrast
    s_dark = bell(dark_ratio, c=0.9869, w=0.06)  # dark pixel fraction

    score = (s_mu * s_sig * s_dark) ** (1 / 3.0)
    return float(np.clip(score, 0.0, 1.0)), float(mu), float(sigma), float(dark_ratio)


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
with tab1:
    uploaded_items = st.file_uploader(
        "📁 Upload images (PNG/JPG/BMP/TIFF) or a folder as .zip",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff", "zip"],
        accept_multiple_files=True
    )

    if uploaded_items:
        results, image_files = [], []
        temp_dir = tempfile.mkdtemp()

        for item in uploaded_items:
            if item.name.lower().endswith(".zip"):
                st.info(f"📦 Extracting ZIP folder: {item.name}")
                with zipfile.ZipFile(io.BytesIO(item.read()), "r") as zf:
                    zf.extractall(temp_dir)
                for root, _, files in os.walk(temp_dir):
                    for f in files:
                        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
                            image_files.append(os.path.join(root, f))
            else:
                image_files.append(item)

        if not image_files:
            st.warning("⚠️ No valid image files found. Please upload PNG, JPG, or BMP images.")
            st.stop()

        st.info(f"🔬 Processing {len(image_files)} image(s)...")
        progress = st.progress(0)

        in_domain, out_domain = 0, 0
        st.session_state["ood_alerts"] = []

        for idx_img, img_item in enumerate(image_files):
            try:
                if isinstance(img_item, str):
                    img = Image.open(img_item).convert("RGB")
                    name = os.path.basename(img_item)
                else:
                    img = Image.open(img_item).convert("RGB")
                    name = img_item.name

                quality_warning = check_image_quality(img)
                domain_score, mu, sigma, dark_ratio = compute_domain_score(img)

                x = transform(img).unsqueeze(0)
                with torch.inference_mode():
                    logits = model(x)
                    probs = torch.softmax(logits, dim=1).squeeze(0)

                conf, idx = torch.max(probs, dim=0)
                pred_label = class_names[idx.item()]
                conf_val = float(conf)

                is_low_conf = conf_val < 0.7
                is_low_domain = domain_score < domain_threshold  # ✅ single source of truth

                warnings_list = []
                if quality_warning:
                    warnings_list.append(quality_warning)
                if conf_val > 0.8 and domain_score < max(0.10, 0.5 * domain_threshold):
                    warnings_list.append(
                        f"High confidence ({conf_val:.2f}) but very low domain score ({domain_score:.2f}) – likely OOD."
                    )

                combined_warning = " | ".join(warnings_list) if warnings_list else ""

                # ✅ Counters use the same logic as the table flag
                if is_low_conf or is_low_domain or quality_warning:
                    out_domain += 1
                else:
                    in_domain += 1

                results.append({
                    "Image": name,
                    "Predicted Class": pred_label,
                    "Confidence": round(conf_val, 4),
                    "Domain Score": round(domain_score, 3),
                    "Preview": img,
                    "Warning": combined_warning,
                    "Low Confidence": is_low_conf,
                    "Low Domain Score": is_low_domain
                })

                # Debug caption to inspect stats vs score
                st.caption(
                    f"{name}: μ={mu:.1f}, σ={sigma:.1f}, dark={dark_ratio:.4f}, score={domain_score:.3f}"
                )

            except Exception as e:
                st.error(f"❌ Error processing {name}: {e}")
            progress.progress((idx_img + 1) / len(image_files))

        progress.empty()

        if st.session_state.get("ood_alerts"):
            if st.session_state["ood_alerts"]:
                st.warning(
                    "⚠️ **Overconfident OOD Predictions:**\n" +
                    "\n".join(st.session_state["ood_alerts"])
                )
                st.session_state["ood_alerts"].clear()

        # ------------------------------------------------------
        # Display results table
        # ------------------------------------------------------
        if results:
            df = pd.DataFrame([{
                "Image": r["Image"],
                "Predicted Class": r["Predicted Class"],
                "Confidence": r["Confidence"],
                "Domain Score": r["Domain Score"],
                "Warning": r["Warning"]
            } for r in results])

            # ✅ Table flag uses the same domain_threshold
            df["Domain Flag"] = df["Domain Score"].apply(
                lambda x: "⚠️ Out-of-domain" if x < domain_threshold else "✅ In-domain"
            )

            st.subheader("📊 Prediction Results")
            styled_df = (
                df.style
                .background_gradient(subset=["Confidence"], cmap="Blues")
                .background_gradient(subset=["Domain Score"], cmap="Oranges")
            )
            st.dataframe(styled_df, use_container_width=True)

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name="karyoassist_predictions.csv",
                mime="text/csv"
            )

            st.markdown("---")
            st.markdown("### 🧠 Domain Check Summary")
            col1, col2 = st.columns(2)
            col1.metric("✅ Likely In-domain", in_domain)
            col2.metric("⚠️ Possibly Out-of-domain", out_domain)

            # ------------------------------------------------------
            # Image previews
            # ------------------------------------------------------
            st.subheader("🖼️ Image Previews")
            cols = st.columns(3)
            for i, r in enumerate(results):
                with cols[i % 3]:
                    caption = f"{r['Image']} → {r['Predicted Class']} ({r['Confidence']:.3f})"
                    if r["Warning"] or r["Low Confidence"] or r["Low Domain Score"]:
                        caption += " ⚠️"
                    st.image(r["Preview"], caption=caption, use_column_width=True)
                    if r["Warning"]:
                        st.warning(r["Warning"])
                    if r["Low Confidence"] or r["Low Domain Score"]:
                        st.warning(
                            f"⚠️ Low confidence ({r['Confidence']:.2f}) or abnormal domain score "
                            f"({r['Domain Score']:.2f}) – likely out-of-domain input."
                        )

            # ------------------------------------------------------
            # Analytics Tab
            # ------------------------------------------------------
            with tab2:
                st.header("📈 Dataset Analytics")

                st.subheader("📊 Class Distribution")
                counts = df["Predicted Class"].value_counts().sort_index()
                st.bar_chart(counts)

                st.subheader("📉 Confidence Histogram")
                fig, ax = plt.subplots()
                ax.hist(df["Confidence"], bins=10, color="skyblue", edgecolor="black")
                ax.set_xlabel("Confidence")
                ax.set_ylabel("Frequency")
                ax.set_title("Confidence Score Distribution")
                st.pyplot(fig)

                st.subheader("🧠 Domain Score Histogram")
                fig2, ax2 = plt.subplots()
                ax2.hist(df["Domain Score"], bins=10, color="orange", edgecolor="black")
                ax2.set_xlabel("Domain Score (higher = in-domain)")
                ax2.set_ylabel("Frequency")
                ax2.set_title("Domain Similarity Distribution")
                st.pyplot(fig2)

                st.subheader("📋 Summary")
                st.write(f"**Total Images:** {len(df)}")
                st.write(f"**Average Confidence:** {df['Confidence'].mean():.4f}")
                st.write(f"**Average Domain Score:** {df['Domain Score'].mean():.4f}")
                st.write(
                    f"**Most Frequent Prediction:** {counts.idxmax()} ({counts.max()} images)"
                )

with tab3:
    st.header("ℹ️ About the Model")
    st.markdown("""
    **Model:** ResNet50 (fine-tuned)  
    **Dataset:** BioImLAB Chromosome Dataset  
    **Classes:** 1–22, X, Y (24 total)  
    **Input Size:** 224 × 224 (RGB)  
    **Training Details:**  
    - Optimizer: Adam  
    - Loss: CrossEntropy  
    - Epochs: Variable (fine-tuned on curated data)  

    **Developed by:** Layan Essam  
    **Purpose:** Automated Karyotyping Assistant to aid cytogenetic analysis in research and diagnostic labs.
    """)

    st.markdown("💡 Future Enhancements:")
    st.markdown("""
    - 🔥 Add Grad-CAM explainability (heatmaps)  
    - 📦 Support larger model architectures (EfficientNet, ViT)  
    - ☁️ Integrate HuggingFace or GDrive model hosting  
    - 📄 Generate downloadable PDF reports  
    - 🧠 Improve OOD detection using feature similarity (embedding space)
    """)
