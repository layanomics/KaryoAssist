
import os, subprocess, pathlib

MODEL_DIR = pathlib.Path("weights")
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "best_resnet50_finetuned.pt"

# Replace with your Google Drive file ID:
FILE_ID = "1zHBz-EMYtM0uTVPd73z48hWG262L7FwZ"

def ensure_model():
    if MODEL_PATH.exists():
        return str(MODEL_PATH)
    subprocess.check_call(["pip", "-q", "install", "gdown"])
    subprocess.check_call(["gdown", f"https://drive.google.com/uc?id={FILE_ID}", "-O", str(MODEL_PATH)])
    return str(MODEL_PATH)
