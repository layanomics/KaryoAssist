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
import json  # for per-class domain profiles

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

# 🔧 Domain sensitivity (fixed, slider removed)
domain_threshold = 0.30

# ----------------------------------------------------------
# Load per-class domain profiles (from training stats)
# ----------------------------------------------------------
DOMAIN_PROFILES = {}
profile_paths = [
    "app/models/domain_profiles.json",  # if you keep it under app/models
    "domain_profiles.json",             # or next to app.py
]

for p in profile_paths:
    if os.path.exists(p):
        try:
            with open(p, "r") as f:
                DOMAIN_PROFILES = json.load(f)
            st.sidebar.success(f"✅ Loaded per-class domain profiles from: {p}")
            break
        except Exception as e:
            st.sidebar.warning(f"⚠️ Failed to load domain_profiles.json from {p}: {e}")

if not DOMAIN_PROFILES:
    st.sidebar.warning("⚠️ domain_profiles.json not found – domain scores will fall back to global heuristics.")

# ----------------------------------------------------------
# Tabs Layout
# ----------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["🔬 Predict", "📈 Analytics", "ℹ️ About Model"])

with tab1:
    st.title("🧬 KaryoAssist")
    st.markdown(
        "Upload **images or an entire folder (.zip)** of chromosome samples to predict their "
        "classes using your fine-tuned **ResNet50** model (trained on BioImLAB dataset)."
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
def check_image_quality(img, min_size=20, min_contrast=5):
    """Return warning string if image looks out-of-domain or poor quality."""
    w, h = img.size
    if w < min_size or h < min_size:
        return "Image too small (likely not a chromosome crop)."
    gray = np.array(img.convert("L"))
    contrast = gray.std()
    if contrast < min_contrast:
        return f"Low contrast (σ={contrast:.1f}) – may be out of domain."
    return None


def _feature_score(val, p25, p50, p75):
    """
    Smooth score in [0,1] based on how close 'val' is to the class-specific distribution.
    Uses p25/p50/p75 from the training stats.
    """
    width = max(p75 - p25, 1e-3)
    return float(np.exp(-((val - p50) / (2 * width)) ** 2))


def compute_domain_score(img, pred_label):
    """
    Per-class domain score based on true training stats.

    - Uses per-class percentiles (p25/p50/p75) for:
      * Mean gray intensity
      * Std gray intensity
      * Dark pixel ratio

    Returns (score, mu, sigma, dark_ratio) where higher score = more in-domain.
    """
    gray = np.array(img.convert("L"), dtype=np.float32)
    mu = float(gray.mean())
    sigma = float(gray.std())
    dark_ratio = float(np.mean(gray < 128))  # dark background → high dark_ratio

    profile_key = f"Class_{pred_label}"
    profile = DOMAIN_PROFILES.get(profile_key)

    # If we don't have a profile (or JSON missing), fall back to global heuristics
    if not profile:
        # Global stats you computed over all training crops
        global_mu_mean, global_mu_std = 17.3889, 11.0052
        global_sigma_mean, global_sigma_std = 33.1041, 14.6327
        global_dark_mean, global_dark_std = 0.9869, 0.0284

        def bell(x, c, s):
            return np.exp(-((x - c) / (2 * s)) ** 2)

        s_mu = bell(mu, global_mu_mean, global_mu_std)
        s_sigma = bell(sigma, global_sigma_mean, global_sigma_std)
        s_dark = bell(dark_ratio, global_dark_mean, global_dark_std)

        score = (s_mu * s_sigma * s_dark) ** (1 / 3)
        return float(np.clip(score, 0.0, 1.0)), mu, sigma, dark_ratio

    mean_prof = profile["Mean"]
    sigma_prof = profile["Sigma"]
    dark_prof = profile["Dark"]

    s_mu = _feature_score(mu, mean_prof["p25"], mean_prof["p50"], mean_prof["p75"])
    s_sigma = _feature_score(sigma, sigma_prof["p25"], sigma_prof["p50"], sigma_prof["p75"])
    s_dark = _feature_score(dark_ratio, dark_prof["p25"], dark_prof["p50"], dark_prof["p75"])

    score = (s_mu * s_sigma * s_dark) ** (1 / 3)
    return float(np.clip(score, 0.0, 1.0)), mu, sigma, dark_ratio


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

        for idx_img, img_item in enumerate(image_files):
            try:
                if isinstance(img_item, str):
                    img = Image.open(img_item).convert("RGB")
                    name = os.path.basename(img_item)
                else:
                    img = Image.open(img_item).convert("RGB")
                    name = img_item.name

                # 1) Basic quality check (size, very low contrast)
                quality_warning = check_image_quality(img)

                # 2) Model prediction
                x = transform(img).unsqueeze(0)
                with torch.inference_mode():
                    logits = model(x)
                    probs = torch.softmax(logits, dim=1).squeeze(0)

                conf, idx = torch.max(probs, dim=0)
                pred_label = class_names[idx.item()]
                conf_val = float(conf)

                # 3) Domain score using *predicted* class statistics
                domain_score, mu, sigma, dark_ratio = compute_domain_score(img, pred_label)

                is_low_conf = conf_val < 0.7
                is_low_domain = domain_score < domain_threshold  # single fixed threshold

                warnings_list = []
                if quality_warning:
                    warnings_list.append(quality_warning)
                if conf_val > 0.8 and is_low_domain:
                    warnings_list.append(
                        f"High confidence ({conf_val:.2f}) but low domain score ({domain_score:.2f}) – likely OOD."
                    )

                combined_warning = " | ".join(warnings_list) if warnings_list else ""

                # ⚠️ Domain counters: ONLY domain score + quality, NOT low confidence
                if is_low_domain or quality_warning:
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
                    "Low Domain Score": is_low_domain,
                    "Mean": round(mu, 2),
                    "Std": round(sigma, 2),
                    "DarkRatio": round(dark_ratio, 4),
                })

                # Optional debug caption (you can remove later)
                st.caption(
                    f"{name} | class={pred_label} | μ={mu:.1f}, σ={sigma:.1f}, "
                    f"dark={dark_ratio:.4f}, score={domain_score:.3f}"
                )

            except Exception as e:
                st.error(f"❌ Error processing {name}: {e}")
            progress.progress((idx_img + 1) / len(image_files))

        progress.empty()

        # ------------------------------------------------------
        # Display results table
        # ------------------------------------------------------
        if results:
            df = pd.DataFrame([{
                "Image": r["Image"],
                "Predicted Class": r["Predicted Class"],
                "Confidence": r["Confidence"],
                "Domain Score": r["Domain Score"],
                "Mean": r["Mean"],
                "Std": r["Std"],
                "DarkRatio": r["DarkRatio"],
                "Warning": r["Warning"],
                "Low Confidence": r["Low Confidence"],
                "Low Domain Score": r["Low Domain Score"],
            } for r in results])

            # ✅ Domain flag uses SAME logic as counters (no low-confidence effect)
            def compute_flag(row):
                has_warning = isinstance(row["Warning"], str) and row["Warning"] != ""
                if row["Low Domain Score"] or has_warning:
                    return "⚠️ Out-of-domain / low quality"
                else:
                    return "✅ In-domain"

            df["Domain Flag"] = df.apply(compute_flag, axis=1)

            st.subheader("📊 Prediction Results")
            styled_df = df.style.background_gradient(subset=["Confidence"], cmap="Blues") \
                                 .background_gradient(subset=["Domain Score"], cmap="Oranges")
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
            # Image previews (small tiles, max 10)
            # ------------------------------------------------------
            st.subheader("🖼️ Image Previews (first 10)")
            max_preview = min(10, len(results))
            cols = st.columns(5)

            for i, r in enumerate(results[:max_preview]):
                with cols[i % 5]:
                    thumb = r["Preview"].resize((180, 180))
                    caption = f"{r['Image']} → {r['Predicted Class']} ({r['Confidence']:.3f})"
                    if r["Warning"] or r["Low Confidence"] or r["Low Domain Score"]:
                        caption += " ⚠️"
                    st.image(thumb, caption=caption, use_column_width=False)
                    
                   # --- Warning Logic for Preview ---
                   if r["Low Domain Score"] or r["Warning"]:
                       # Domain OR quality → Real warning
                       st.warning(
                            f"⚠️ Low domain score ({r['Domain Score']:.2f}) or poor quality – possibly out-of-domain input."
                       )
                   elif r["Low Confidence"]:
                       # Confidence ONLY → Soft warning (NOT OOD)
                       st.info(
                          f"ℹ️ Low confidence ({r['Confidence']:.2f}) – prediction uncertain but image is in-domain."
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
                ax.hist(df["Confidence"], bins=10)
                ax.set_xlabel("Confidence")
                ax.set_ylabel("Frequency")
                ax.set_title("Confidence Score Distribution")
                st.pyplot(fig)

                st.subheader("🧠 Domain Score Histogram")
                fig2, ax2 = plt.subplots()
                ax2.hist(df["Domain Score"], bins=10)
                ax2.set_xlabel("Domain Score (higher = in-domain)")
                ax2.set_ylabel("Frequency")
                ax2.set_title("Domain Similarity Distribution")
                st.pyplot(fig2)

                st.subheader("📋 Summary")
                st.write(f"**Total Images:** {len(df)}")
                st.write(f"**Average Confidence:** {df['Confidence'].mean():.4f}")
                st.write(f"**Average Domain Score:** {df['Domain Score'].mean():.4f}")
                st.write(f"**Most Frequent Prediction:** {counts.idxmax()} ({counts.max()} images)")

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
