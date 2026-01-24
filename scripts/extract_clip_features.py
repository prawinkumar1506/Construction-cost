import torch
import clip
import rasterio
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from PIL import Image

BASE = Path("d:/Projects/Construction-cost")
DATA_DIR = BASE / "datasets"
TRAIN_IMG_DIR = DATA_DIR / "train_composite"
EVAL_IMG_DIR  = DATA_DIR / "evaluate_composite"

CACHE_DIR = BASE / "cache" / "clip_feats"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# Load CLIP
model, preprocess = clip.load("ViT-B/32", device=DEVICE)
model.eval()

def find_image(fname):
    p1 = TRAIN_IMG_DIR / fname
    if p1.exists():
        return p1
    p2 = EVAL_IMG_DIR / fname
    if p2.exists():
        return p2
    return None

def load_image_safe(path):
    with rasterio.open(path) as ds:
        arr = ds.read()  # (bands, H, W)

    # Use RGB-like bands if available
    if arr.shape[0] >= 3:
        img = np.stack([arr[2], arr[1], arr[0]], axis=-1)
    else:
        img = np.repeat(arr[0:1], 3, axis=0).transpose(1,2,0)

    img = img.astype(np.float32)

    # Robust contrast stretch
    p1, p99 = np.percentile(img, (1, 99))
    if p99 > p1:
        img = (img - p1) / (p99 - p1)
    else:
        img = img - img.min()
        img = img / (img.max() + 1e-6)

    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)

    return Image.fromarray(img)

@torch.no_grad()
def extract_feat(pil_img):
    image = preprocess(pil_img).unsqueeze(0).to(DEVICE)
    feat = model.encode_image(image)
    feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().numpy().ravel()

def main():
    train = pd.read_csv(DATA_DIR / "train_tabular.csv")
    eval_df = pd.read_csv(DATA_DIR / "evaluation_tabular_no_target.csv")
    all_df = pd.concat([train, eval_df], ignore_index=True)

    ok, fail, missing = 0, 0, 0

    for _, row in tqdm(all_df.iterrows(), total=len(all_df)):
        fname = row["sentinel2_tiff_file_name"]
        key = Path(fname).stem
        out = CACHE_DIR / f"{key}.npy"

        if out.exists():
            continue

        path = find_image(fname)
        if path is None:
            print("MISSING FILE:", fname)
            np.save(out, np.zeros(512, dtype=np.float32))
            missing += 1
            continue

        try:
            pil_img = load_image_safe(path)
            feat = extract_feat(pil_img)

            if np.isnan(feat).any() or np.abs(feat).sum() < 1e-6:
                raise RuntimeError("Invalid CLIP feature")

            np.save(out, feat.astype(np.float32))
            ok += 1

        except Exception as e:
            print("FAILED:", fname, e)
            np.save(out, np.zeros(512, dtype=np.float32))
            fail += 1

    print("\n==============================")
    print("CLIP extraction finished")
    print("OK     :", ok)
    print("FAIL   :", fail)
    print("MISSING:", missing)
    print("==============================")

if __name__ == "__main__":
    main()
