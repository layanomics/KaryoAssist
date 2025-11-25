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
import json
import base64
from streamlit_sortables import sort_items  # 🔹 for drag-and-drop ordering

# ----------------------------------------------------------
# Streamlit Page Config
# ----------------------------------------------------------
st.set_page_config(page_title="KaryoAssist", page_icon="🧬", layout="wide")

# ----------------------------------------------------------
# Initialize dynamic uploader key (so Clear button works)
# ----------------------------------------------------------
if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = "uploader_1"

# Sidebar
st.sidebar.title("⚙️ Settings & Info")
st.sidebar.markdown("""
**KaryoAssist** — an AI-powered assistant for automated chromosome classification.  
Upload single images, multiple images, or a `.zip` folder for batch prediction.
""")
st.sidebar.info("Model: Fine-tuned **ResNet50** (24 chromosome classes).")

# Fixed domain threshold
domain_threshold = 0.30

# ----------------------------------------------------------
# Load per-class domain profiles
# ----------------------------------------------------------
DOMAIN_PROFILES = {}
profile_paths = ["app/models/domain_profiles.json", "domain_profiles.json"]

for p in profile_paths:
    if os.path.exists(p):
        try:
            with open(p, "r") as f:
                DOMAIN_PROFILES = json.load(f)
            st.sidebar.success(f"Loaded domain profiles from: {p}")
            break
        except Exception as e:
            st.sidebar.warning(f"Failed to load {p}: {e}")

if not DOMAIN_PROFILES:
    st.sidebar.warning("domain_profiles.json not found — falling back to global heuristics.")

# ----------------------------------------------------------
# Tabs Layout
# ----------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["🔬 Predict", "📈 Analytics", "ℹ️ About Model"])

# ----------------------------------------------------------
# Predict Tab
# ----------------------------------------------------------
with tab1:
    st.title("🧬 KaryoAssist")
    st.markdown("Upload images or a `.zip` folder to classify chromosomes.")

    # ---------------------- CLEAR BUTTON ----------------------
    if st.button("🧹 Clear Previous Analysis"):
        st.session_state.clear()
        new_key = f"uploader_{np.random.randint(1, 1_000_000)}"
        st.session_state["uploader_key"] = new_key
        st.rerun()
    # ----------------------------------------------------------

# ----------------------------------------------------------
# Load Model
# ----------------------------------------------------------
@st.cache_resource
def load_model():
    model_path = "app/models/best_resnet50_finetuned.ckpt"

    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
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
    st.sidebar.success("Model loaded successfully")

    return model, class_names

model, class_names = load_model()

# ----------------------------------------------------------
# Image Quality / Domain Check
# ----------------------------------------------------------
def check_image_quality(img, min_size=20, min_contrast=5):
    w, h = img.size
    if w < min_size or h < min_size:
        return "Image too small."
    gray = np.array(img.convert("L"))
    if gray.std() < min_contrast:
        return f"Low contrast (σ={gray.std():.1f})"
    return None

def _feature_score(val, p25, p50, p75):
    width = max(p75 - p25, 1e-3)
    return float(np.exp(-((val - p50) / (2 * width)) ** 2))

def compute_domain_score(img, pred_label):
    gray = np.array(img.convert("L"), dtype=np.float32)
    mu = float(gray.mean())
    sigma = float(gray.std())
    dark_ratio = float(np.mean(gray < 128))

    profile_key = f"Class_{pred_label}"
    profile = DOMAIN_PROFILES.get(profile_key)

    if not profile:
        global_mu_mean, global_mu_std = 17.39, 11.01
        global_sigma_mean, global_sigma_std = 33.10, 14.63
        global_dark_mean, global_dark_std = 0.987, 0.028

        def bell(x, c, s):
            return np.exp(-((x - c) / (2 * s)) ** 2)

        s_mu = bell(mu, global_mu_mean, global_mu_std)
        s_sigma = bell(sigma, global_sigma_mean, global_sigma_std)
        s_dark = bell(dark_ratio, global_dark_mean, global_dark_std)

        score = (s_mu * s_sigma * s_dark) ** (1 / 3)
        return float(np.clip(score, 0, 1)), mu, sigma, dark_ratio

    mean_prof = profile["Mean"]
    sigma_prof = profile["Sigma"]
    dark_prof = profile["Dark"]

    s_mu = _feature_score(mu, mean_prof["p25"], mean_prof["p50"], mean_prof["p75"])
    s_sigma = _feature_score(sigma, sigma_prof["p25"], sigma_prof["p50"], sigma_prof["p75"])
    s_dark = _feature_score(dark_ratio, dark_prof["p25"], dark_prof["p50"], dark_prof["p75"])

    score = (s_mu * s_sigma * s_dark) ** (1 / 3)
    return float(np.clip(score, 0, 1)), mu, sigma, dark_ratio

# ----------------------------------------------------------
# Preprocessing
# ----------------------------------------------------------
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Utility for sorting previews according to karyogram order
def karyo_order(label):
    if label == "X":
        return 23
    if label == "Y":
        return 24
    return int(label)

# ----------------------------------------------------------
# File Upload Processing
# ----------------------------------------------------------
with tab1:
    uploaded_items = st.file_uploader(
        "📁 Upload chromosome images or zip folder",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff", "zip"],
        accept_multiple_files=True,
        key=st.session_state["uploader_key"]
    )

    if uploaded_items:
        results = []
        image_files = []
        temp_dir = tempfile.mkdtemp()

        # ---------- De-duplicate images within this run ----------
        seen_keys = set()

        for item in uploaded_items:
            if item.name.lower().endswith(".zip"):
                with zipfile.ZipFile(io.BytesIO(item.read()), "r") as zf:
                    zf.extractall(temp_dir)
                for root, _, files in os.walk(temp_dir):
                    for f in files:
                        if f.lower().endswith(("png", "jpg", "jpeg", "bmp", "tif", "tiff")):
                            full_path = os.path.join(root, f)
                            # Use relative path inside temp_dir as a stable key
                            rel_key = os.path.relpath(full_path, temp_dir)
                            if rel_key in seen_keys:
                                continue
                            seen_keys.add(rel_key)
                            image_files.append(full_path)
            else:
                # Directly uploaded images: use filename as key
                key = item.name
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                image_files.append(item)
        # ---------------------------------------------------------

        st.info(f"Processing {len(image_files)} image(s)...")
        progress = st.progress(0)

        in_domain = 0
        out_domain = 0

        for idx_img, img_item in enumerate(image_files):
            try:
                if isinstance(img_item, str):
                    img = Image.open(img_item).convert("RGB")
                    name = os.path.basename(img_item)
                else:
                    img = Image.open(img_item).convert("RGB")
                    name = img_item.name

                quality_warning = check_image_quality(img)

                x = transform(img).unsqueeze(0)
                with torch.inference_mode():
                    logits = model(x)
                    probs = torch.softmax(logits, 1).squeeze(0)

                conf_val, idx = torch.max(probs, 0)
                conf_val = float(conf_val)
                pred_label = class_names[idx.item()]

                domain_score, mu, sigma, dark_ratio = compute_domain_score(img, pred_label)

                is_low_conf = conf_val < 0.55
                is_low_domain = domain_score < domain_threshold

                warning_text = ""
                if quality_warning:
                    warning_text = quality_warning
                if conf_val > 0.8 and is_low_domain:
                    warning_text = f"High confidence but low domain score ({domain_score:.2f})"

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
                    "Warning": warning_text,
                    "Low Confidence": is_low_conf,
                    "Low Domain Score": is_low_domain,
                    "Mean": round(mu, 2),
                    "Std": round(sigma, 2),
                    "DarkRatio": round(dark_ratio, 4),
                })

            except Exception as e:
                st.error(f"Error processing {name}: {e}")

            progress.progress((idx_img + 1) / len(image_files))

        progress.empty()

        # ------------------------------------------------------
        # Display Table
        # ------------------------------------------------------
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

        df.index = df.index + 1

        def compute_flag(row):
            has_warning = isinstance(row["Warning"], str) and row["Warning"] != ""
            if row["Low Domain Score"] or has_warning:
                return "⚠️ Out-of-domain / low quality"
            else:
                return "✅ In-domain"

        df["Domain Flag"] = df.apply(compute_flag, axis=1)

        st.subheader("📊 Prediction Results")
        st.dataframe(
            df.style.background_gradient(subset=["Confidence"], cmap="Blues")
                     .background_gradient(subset=["Domain Score"], cmap="Oranges"),
            use_container_width=True
        )

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download CSV", data=csv, file_name="karyoassist_predictions.csv")

        # ------------------------------------------------------
        # Domain Summary
        # ------------------------------------------------------
        st.markdown("---")
        st.subheader("🧠 Domain Check Summary")
        c1, c2 = st.columns(2)
        c1.metric("Likely In-domain", in_domain)
        c2.metric("Possibly Out-of-domain", out_domain)

        # ------------------------------------------------------
        # Image Previews (KARYOGRAM ORDER + DRAG-AND-DROP)
        # ------------------------------------------------------
        st.subheader("🖼️ Image Previews (up to 46, karyogram order)")

        # Sort the results by karyogram order before preview
        results_sorted = sorted(results, key=lambda r: karyo_order(r["Predicted Class"]))

        max_preview = min(46, len(results_sorted))
        preview_results = results_sorted[:max_preview]

        if len(results_sorted) > max_preview:
            st.caption(
                f"Showing first {max_preview} images for preview and reordering. "
                "Additional images are still included in the table, CSV, and analytics."
            )

        # ---- Drag-and-drop reordering using streamlit-sortables ----
        # Build HTML cards (image + label + light-blue background)
        items = []
        for i, r in enumerate(preview_results):
            # Convert thumbnail to base64
            buf = io.BytesIO()
            thumb_img = r["Preview"].resize((180, 180))
            thumb_img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            warning_line = ""
            if r["Warning"] or r["Low Confidence"] or r["Low Domain Score"]:
                warning_line = (
                    "<div style='color:#b00020;font-size:11px;margin-top:4px;'>"
                    "⚠ Check domain / confidence"
                    "</div>"
                )

            card_html = f"""
            <div style="
                background-color:#e9f4ff;
                border-radius:12px;
                padding:8px;
                text-align:center;
                box-sizing:border-box;
            ">
                <img src="data:image/png;base64,{b64}"
                     style="width:120px;border-radius:8px;"/>
                <div style="font-size:13px;font-weight:600;margin-top:4px;">
                    {r['Predicted Class']} ({r['Confidence']:.3f})
                </div>
                <div style="font-size:11px;color:#555;">
                    {r['Image']}
                </div>
                {warning_line}
            </div>
            """
            # Prefix with index so we can recover order later
            items.append(f"{i}:: {card_html}")

        st.markdown("**Drag the cards below to change the display order of the previews:**")
        sorted_items = sort_items(items)

        # Recover the new order of indices from the sorted strings
        order_indices = []
        for s_item in sorted_items:
            try:
                idx_str = s_item.split("::", 1)[0]
                order_indices.append(int(idx_str))
            except Exception:
                continue

        # Fallback: if parsing failed for some reason, keep original order
        if len(order_indices) != len(preview_results):
            ordered_preview_results = preview_results
        else:
            ordered_preview_results = [preview_results[i] for i in order_indices]

        # ---- Show the thumbnails in the user-defined order (5-column grid) ----
        cols = st.columns(5)

        for i, r in enumerate(ordered_preview_results):
            with cols[i % 5]:
                thumb = r["Preview"].resize((180, 180))
                caption = f"{r['Image']} → {r['Predicted Class']} ({r['Confidence']:.3f})"
                if r["Warning"] or r["Low Confidence"] or r["Low Domain Score"]:
                    caption += " ⚠️"
                st.image(thumb, caption=caption, use_column_width=False)

                if r["Low Domain Score"] or r["Warning"]:
                    st.warning(
                        f"⚠️ Low domain score ({r['Domain Score']:.2f}) or poor quality – possibly out-of-domain."
                    )
                elif r["Low Confidence"]:
                    st.info(
                        f"ℹ️ Low confidence ({r['Confidence']:.2f}) – uncertain but in-domain."
                    )

# ----------------------------------------------------------
# Analytics Tab
# ----------------------------------------------------------
with tab2:
    st.header("📈 Dataset Analytics")

    if "df" not in locals() and "df" not in globals():
        st.info("Run predictions in the Predict tab to generate analytics.")
        st.stop()

    st.subheader("Class Distribution")

    # ----- Sort classes in true karyogram order: 1..22, X, Y -----
    class_order = [str(i) for i in range(1, 23)] + ["X", "Y"]
    cat_classes = pd.Categorical(df["Predicted Class"], categories=class_order, ordered=True)
    class_counts = pd.value_counts(cat_classes).sort_index()
    st.bar_chart(class_counts)
    # -------------------------------------------------------------

    st.subheader("Confidence Histogram")
    fig, ax = plt.subplots()
    ax.hist(df["Confidence"], bins=10)
    st.pyplot(fig)

    st.subheader("Domain Score Histogram")
    fig2, ax2 = plt.subplots()
    ax2.hist(df["Domain Score"], bins=10)
    st.pyplot(fig2)

    st.subheader("Summary")
    st.write(f"Total Images: {len(df)}")
    st.write(f"Average Confidence: {df['Confidence'].mean():.4f}")
    st.write(f"Average Domain Score: {df['Domain Score'].mean():.4f}")

# ----------------------------------------------------------
# About Tab
# ----------------------------------------------------------
with tab3:
    st.header("ℹ️ About the Model")
    st.markdown("""
    **Model:** ResNet50 (fine-tuned)  
    **Dataset:** BioImLAB Chromosome Dataset  
    **Classes:** 1–22, X, Y  
    **Purpose:** Automated chromosome classification for cytogenetics.
    """)
