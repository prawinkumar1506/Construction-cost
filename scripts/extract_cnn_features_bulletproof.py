"""
🔥 BULLETPROOF CNN FEATURE EXTRACTION
Fixes all issues from previous versions.

Guarantees:
✅ Loads TIFFs from correct path (train_composite / evaluation_composite)
✅ Handles NaN values robustly
✅ Normalizes images safely
✅ Reports all failures clearly
✅ Checks for invalid features
✅ CUDA if available, CPU fallback
"""

import torch
import torchvision.transforms as T
import torchvision.models as models
import rasterio
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

BASE = Path("d:/Projects/Construction-cost")
DATA_DIR = BASE / "datasets"
TRAIN_COMPOSITE = DATA_DIR / "train_composite"
EVAL_COMPOSITE = DATA_DIR / "evaluation_composite"
CACHE_DIR = BASE / "cache" / "cnn_feats"

# Create cache
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Device: {DEVICE}")

# Load ResNet50
print("📥 Loading ResNet50...")
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.fc = torch.nn.Identity()
model = model.to(DEVICE)
model.eval()
print("✅ ResNet50 loaded")

transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_image_safe(path):
    """Load image robustly, handling NaN and edge cases"""
    with rasterio.open(path) as ds:
        arr = ds.read()  # (bands, H, W)
    
    # Handle NaN values
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    
    # Use first 3 bands safely
    if arr.shape[0] >= 3:
        img = np.stack([arr[2], arr[1], arr[0]], axis=-1)  # RGB
    else:
        img = np.repeat(arr[0:1], 3, axis=0).transpose(1, 2, 0)
    
    # Normalize robustly
    img = img.astype(np.float32)
    
    # Use percentile normalization (robust to outliers)
    p1, p99 = np.percentile(img, (1, 99))
    if p99 > p1 + 1e-6:
        img = (img - p1) / (p99 - p1)
    else:
        img = img - img.min()
        if img.max() > 1e-6:
            img = img / img.max()
    
    # Clip and scale to [0, 255]
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    
    return img

@torch.no_grad()
def extract_feat(img):
    """Extract ResNet50 features"""
    x = transform(img).unsqueeze(0).to(DEVICE)
    f = model(x).cpu().numpy().ravel()
    return f

def extract_with_validation(path):
    """Extract feature with validation"""
    try:
        img = load_image_safe(path)
        feat = extract_feat(img)
        
        # Validate feature
        if np.isnan(feat).any():
            return None, "NaN in feature"
        if np.isinf(feat).any():
            return None, "Inf in feature"
        if np.abs(feat).sum() < 1e-6:
            return None, "Feature all zeros"
        
        return feat, None
    except Exception as e:
        return None, str(e)

def main():
    print(f"\n{'='*70}")
    print("🚀 CNN FEATURE EXTRACTION")
    print(f"{'='*70}\n")
    
    # Load metadata
    train = pd.read_csv(DATA_DIR / "train_tabular.csv")
    eval_df = pd.read_csv(DATA_DIR / "evaluation_tabular_no_target.csv")
    all_df = pd.concat([train, eval_df], ignore_index=True)
    
    print(f"Processing {len(all_df)} images\n")
    
    ok = 0
    fail = 0
    skipped = 0
    failed_files = []
    
    for idx, (_, row) in enumerate(tqdm(all_df.iterrows(), total=len(all_df), desc="Extracting")):
        fname = row["sentinel2_tiff_file_name"]
        key = Path(fname).stem
        out = CACHE_DIR / f"{key}.npy"
        
        # Skip if already exists
        if out.exists():
            skipped += 1
            continue
        
        # Determine which folder
        path_train = TRAIN_COMPOSITE / fname
        path_eval = EVAL_COMPOSITE / fname
        
        path = None
        if path_train.exists():
            path = path_train
        elif path_eval.exists():
            path = path_eval
        else:
            # File not found
            np.save(out, np.zeros(2048, dtype=np.float32))
            fail += 1
            failed_files.append((fname, "File not found"))
            continue
        
        # Extract
        feat, error = extract_with_validation(path)
        
        if feat is not None:
            np.save(out, feat)
            ok += 1
        else:
            np.save(out, np.zeros(2048, dtype=np.float32))
            fail += 1
            failed_files.append((fname, error))
    
    print(f"\n{'='*70}")
    print("✅ EXTRACTION COMPLETE")
    print(f"{'='*70}\n")
    
    print(f"OK       : {ok:4d}")
    print(f"FAIL     : {fail:4d}")
    print(f"SKIPPED  : {skipped:4d}")
    print(f"TOTAL    : {ok + fail + skipped:4d}\n")
    
    # Show failures
    if failed_files:
        print(f"Failed files (first 10):")
        for fname, error in failed_files[:10]:
            print(f"  ❌ {fname}: {error}")
    
    # Verify cache
    print(f"\n{'='*70}")
    print("📊 CACHE VERIFICATION")
    print(f"{'='*70}\n")
    
    cache_files = list(CACHE_DIR.glob("*.npy"))
    print(f"Total cache files: {len(cache_files)}")
    
    zero_count = 0
    for f in cache_files:
        x = np.load(f)
        if np.abs(x).sum() < 1e-6:
            zero_count += 1
    
    valid_count = len(cache_files) - zero_count
    print(f"Valid features:    {valid_count}")
    print(f"Zero vectors:      {zero_count}")
    print(f"Success rate:      {100 * valid_count / len(cache_files):.1f}%")
    
    print(f"\n{'='*70}\n")
    
    if valid_count >= 1800:
        print("✅ SUCCESS! Cache is ready for training")
        return True
    else:
        print(f"⚠️  WARNING: Only {valid_count}/2048 valid features")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
