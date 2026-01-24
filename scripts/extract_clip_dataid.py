import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
import clip
import rasterio
import cv2

# ================= CONFIG =================
TRAIN_CSV = "datasets/train_tabular.csv"
EVAL_CSV  = "datasets/evaluation_tabular_no_target.csv"

TRAIN_IMG_DIR = "datasets/train_composite"
EVAL_IMG_DIR  = "datasets/evaluation_composite"

OUT_DIR = "cache/clip_feats"
os.makedirs(OUT_DIR, exist_ok=True)

BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", DEVICE)

# ================= LOAD CLIP =================
model, preprocess = clip.load("ViT-B/32", device=DEVICE)
model.eval()

# ================= LOAD TABLE =================
train = pd.read_csv(TRAIN_CSV)
evald = pd.read_csv(EVAL_CSV)
all_df = pd.concat([train, evald], ignore_index=True)

print("Total rows:", len(all_df))

# ================= HELPERS =================
def find_image(row):
    fname_s2 = row.get("sentinel2_tiff_file_name", None)
    if pd.isna(fname_s2):
        return None

    if row.name < len(train):
        path = os.path.join(TRAIN_IMG_DIR, fname_s2)
    else:
        path = os.path.join(EVAL_IMG_DIR, fname_s2)

    if os.path.exists(path):
        return path
    return None


def load_sentinel_rgb(path):
    with rasterio.open(path) as src:
        img = src.read()  # (C, H, W)

    if img.shape[0] < 3:
        raise ValueError("Not enough bands")

    rgb = img[:3]
    rgb = np.transpose(rgb, (1, 2, 0))
    rgb = np.nan_to_num(rgb)

    # Normalize to 0-255
    rgb = rgb - rgb.min()
    if rgb.max() > 0:
        rgb = rgb / rgb.max()

    rgb = (rgb * 255).astype(np.uint8)
    rgb = cv2.resize(rgb, (224, 224))

    return Image.fromarray(rgb)


# ================= PROCESS =================
features = {}
failed = 0

batch_imgs = []
batch_ids = []

def flush_batch():
    global batch_imgs, batch_ids
    if len(batch_imgs) == 0:
        return

    with torch.no_grad():
        imgs = torch.stack([preprocess(im) for im in batch_imgs]).to(DEVICE)
        feats = model.encode_image(imgs)
        feats = feats.cpu().numpy()

    for did, f in zip(batch_ids, feats):
        np.save(os.path.join(OUT_DIR, f"{did}.npy"), f)

    batch_imgs = []
    batch_ids = []


for idx, row in tqdm(all_df.iterrows(), total=len(all_df)):
    data_id = row["data_id"]

    out_path = os.path.join(OUT_DIR, f"{data_id}.npy")
    if os.path.exists(out_path):
        continue

    try:
        img_path = find_image(row)
        if img_path is None:
            raise ValueError("Image not found")

        img = load_sentinel_rgb(img_path)

        batch_imgs.append(img)
        batch_ids.append(data_id)

        if len(batch_imgs) >= BATCH_SIZE:
            flush_batch()

    except Exception as e:
        failed += 1
        print("❌ Failed:", data_id, str(e))

flush_batch()

print("===================================")
print("CLIP extraction finished")
print("Saved:", len(os.listdir(OUT_DIR)))
print("Failed:", failed)
print("Output dir:", OUT_DIR)
print("===================================")
