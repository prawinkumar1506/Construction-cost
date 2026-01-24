import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import re

# ==============================
# CONFIG
# ==============================
TRAIN_CSV = "datasets/train_tabular.csv"
TEST_CSV  = "datasets/evaluation_tabular_no_target.csv"
CLIP_DIR = "cache/clip_feats"
OUT_DIR = "cache/clip_row_feats"

os.makedirs(OUT_DIR, exist_ok=True)

# ==============================
# HELPERS
# ==============================
def normalize_location(name: str):
    """
    Convert:
      '29000 Nara' -> 'nara'
      '05000 Akita' -> 'akita'
      'Dinagat Islands' -> 'dinagat_islands'
    """
    name = name.lower().strip()
    name = re.sub(r"^\d+\s*", "", name)  # remove leading codes
    name = name.replace(" ", "_")
    return name

def parse_clip_filename(fname):
    # sentinel_2_01000_hokkaido_2019-Q1.npy
    # -> location = hokkaido, quarter = 2019-Q1
    base = os.path.basename(fname)
    parts = base.replace(".npy","").split("_")
    location = parts[3]
    quarter = parts[3]
    return location.lower(), quarter

# ==============================
# LOAD DATA
# ==============================
print("Loading datasets...")
train = pd.read_csv(TRAIN_CSV)
test  = pd.read_csv(TEST_CSV)

full = pd.concat([
    train[["data_id","geolocation_name","quarter_label"]],
    test[["data_id","geolocation_name","quarter_label"]]
], ignore_index=True)

full["norm_loc"] = full["geolocation_name"].apply(normalize_location)

print("Example normalized locations:")
print(full[["geolocation_name","norm_loc"]].head(10))

# ==============================
# INDEX CLIP FILES
# ==============================
print("Indexing CLIP files...")
clip_index = {}

for fname in os.listdir(CLIP_DIR):
    if not fname.endswith(".npy"):
        continue
    loc, qtr = parse_clip_filename(fname)
    key = (loc, qtr)
    if key not in clip_index:
        clip_index[key] = []
    clip_index[key].append(os.path.join(CLIP_DIR, fname))

print("Total unique (loc,quarter) in CLIP:", len(clip_index))

# ==============================
# BUILD ROW FEATURES
# ==============================
print("Building row-level CLIP features...")

found = 0
missing = 0

for _, row in tqdm(full.iterrows(), total=len(full)):
    data_id = row["data_id"]
    loc = row["norm_loc"]
    qtr = row["quarter_label"]

    key = (loc, qtr)

    out_path = os.path.join(OUT_DIR, f"{data_id}.npy")

    if key not in clip_index:
        missing += 1
        continue

    feats = []
    for f in clip_index[key]:
        v = np.load(f)
        if np.isnan(v).any():
            continue
        feats.append(v)

    if len(feats) == 0:
        missing += 1
        continue

    feats = np.stack(feats)
    mean_feat = feats.mean(axis=0)

    np.save(out_path, mean_feat)
    found += 1

print("="*50)
print("DONE")
print("Found:", found)
print("Missing:", missing)
print("Total:", len(full))
print("="*50)

# ==============================
# VERIFY
# ==============================
lens = []
for f in os.listdir(OUT_DIR):
    v = np.load(os.path.join(OUT_DIR,f))
    lens.append(len(v))

print("Embedding dim example:", lens[:5])
print("Unique embedding dims:", set(lens))
