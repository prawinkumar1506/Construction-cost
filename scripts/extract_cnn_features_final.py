#!/usr/bin/env python3
"""Extract CNN features using simpler approach to avoid torch initialization issues"""

import os
os.environ['TORCH_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import torch
torch.set_num_threads(1)
import torch.nn as nn

print("✓ Starting CNN feature extraction...")

import torchvision.transforms as T
import torchvision.models as models
import rasterio
import numpy as np
import pandas as pd
from pathlib import Path
import gc

BASE = Path("d:/Projects/Construction-cost")
DATA_DIR = BASE / "datasets"
CACHE_DIR = BASE / "cache" / "cnn_feats"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cpu"  # Force CPU to avoid GPU initialization issues
print(f"✓ Device: {DEVICE}")

# Load pretrained ResNet50 ONCE
print("✓ Loading ResNet50 (this may take a minute)...")
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.fc = nn.Identity()
model.eval()
print("✓ ResNet50 loaded")

transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

def load_sentinel(path):
    """Load satellite image from GeoTIFF"""
    with rasterio.open(path) as ds:
        arr = ds.read()
    if arr.shape[0] >= 3:
        img = np.stack([arr[2], arr[1], arr[0]], axis=-1)
    else:
        img = np.repeat(arr[0:1], 3, axis=0).transpose(1,2,0)
    img = img.astype(np.float32)
    if img.size > 0:
        img = img - img.min()
        img = img / (img.max() + 1e-6)
    img = (img * 255).astype(np.uint8)
    return img

@torch.no_grad()
def extract_feature(img):
    x = transform(img).unsqueeze(0)
    feat = model(x).numpy().ravel()
    return feat

# Load metadata
print("✓ Loading metadata...")
train = pd.read_csv(DATA_DIR / "train_tabular.csv")
eval_df = pd.read_csv(DATA_DIR / "evaluation_tabular_no_target.csv")
all_df = pd.concat([train, eval_df], ignore_index=True)

print(f"✓ Processing {len(all_df)} images...\n")

errors = 0
success = 0
skipped = 0

for idx, (_, row) in enumerate(all_df.iterrows()):
    fname = row["sentinel2_tiff_file_name"]
    key = Path(fname).stem
    cache_file = CACHE_DIR / f"{key}.npy"

    if cache_file.exists():
        skipped += 1
        continue

    # Check both composite folders
    img_path_train = DATA_DIR / "train_composite" / fname
    img_path_eval = DATA_DIR / "evaluation_composite" / fname

    img_path = None
    if img_path_train.exists():
        img_path = img_path_train
    elif img_path_eval.exists():
        img_path = img_path_eval

    if img_path is None:
        feat = np.zeros(2048)
        errors += 1
    else:
        try:
            img = load_sentinel(img_path)
            feat = extract_feature(img)
            success += 1
        except Exception as e:
            print(f"  Error {fname}: {str(e)[:50]}")
            feat = np.zeros(2048)
            errors += 1

    np.save(cache_file, feat)
    gc.collect()

    if (idx + 1) % 100 == 0:
        total_cached = len(list(CACHE_DIR.glob('*.npy')))
        print(f"  {idx + 1:4d}/{len(all_df)} | Cached: {total_cached:4d} | Success: {success:4d} | Errors: {errors:2d} | Skipped: {skipped:4d}")

print(f"\n{'='*60}")
print(f"✓ COMPLETE")
print(f"  Total images: {len(all_df)}")
print(f"  New features extracted: {success}")
print(f"  Skipped (cached): {skipped}")
print(f"  Errors: {errors}")
print(f"  Total files in cache: {len(list(CACHE_DIR.glob('*.npy')))}")
print(f"{'='*60}\n")
