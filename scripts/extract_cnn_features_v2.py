#!/usr/bin/env python3
"""Extract CNN features from Sentinel images using ResNet50"""

def main():
    import torch
    torch.set_num_threads(4)
    
    import torchvision.transforms as T
    import torchvision.models as models
    import rasterio
    import numpy as np
    import pandas as pd
    from pathlib import Path
    from tqdm import tqdm

    BASE = Path("d:/Projects/Construction-cost")
    DATA_DIR = BASE / "datasets"
    CACHE_DIR = BASE / "cache" / "cnn_feats"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    # Load pretrained ResNet50
    print("Loading ResNet50...")
    model = models.resnet50(pretrained=True)
    model.fc = torch.nn.Identity()
    model = model.to(DEVICE)
    model.eval()

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((224,224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    def load_sentinel(path):
        with rasterio.open(path) as ds:
            arr = ds.read()
        if arr.shape[0] >= 3:
            img = np.stack([arr[2], arr[1], arr[0]], axis=-1)
        else:
            img = np.repeat(arr[0:1], 3, axis=0).transpose(1,2,0)
        img = img.astype(np.float32)
        img = img - img.min()
        img = img / (img.max() + 1e-6)
        img = (img * 255).astype(np.uint8)
        return img

    @torch.no_grad()
    def extract_feature(img):
        x = transform(img).unsqueeze(0).to(DEVICE)
        feat = model(x).cpu().numpy().ravel()
        return feat

    # Load metadata
    train = pd.read_csv(DATA_DIR / "train_tabular.csv")
    eval_df = pd.read_csv(DATA_DIR / "evaluation_tabular_no_target.csv")
    all_df = pd.concat([train, eval_df], ignore_index=True)

    print(f"Processing {len(all_df)} images...")

    for idx, (_, row) in enumerate(all_df.iterrows()):
        fname = row["sentinel2_tiff_file_name"]
        key = Path(fname).stem
        cache_file = CACHE_DIR / f"{key}.npy"

        if cache_file.exists():
            continue

        try:
            img = load_sentinel(DATA_DIR / fname)
            feat = extract_feature(img)
        except Exception as e:
            print(f"Error: {fname} - {e}")
            feat = np.zeros(2048)

        np.save(cache_file, feat)
        
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(all_df)}")

    print("✓ Done extracting CNN features.")

if __name__ == "__main__":
    main()
