"""
🚀 GPU Feature Extraction using PyTorch (Fast & Simple)
Extracts features in MINUTES instead of HOURS
"""
import numpy as np
import pandas as pd
from pathlib import Path
import rasterio
import torch
from tqdm.auto import tqdm

BASE = Path('d:/Projects/Construction-cost')
DATA_DIR = BASE / 'datasets'

# Check GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('='*60)
if torch.cuda.is_available():
    print(f'✅ GPU DETECTED: {torch.cuda.get_device_name(0)}')
    print(f'✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    print('✅ Using GPU for feature extraction')
else:
    print('⚠️ GPU NOT FOUND - Using CPU (slow)')
print('='*60)

def extract_features_gpu(tif_path):
    """GPU-accelerated feature extraction with CORRECT Sentinel-2 bands + ADVANCED features"""
    try:
        with rasterio.open(tif_path) as src:
            bands = src.read()
            
            if bands.shape[0] >= 11:  # Sentinel-2 (12 bands)
                # ✅ CORRECT Sentinel-2 band mapping
                blue = torch.from_numpy(bands[1]).float().to(device)   # B2
                green = torch.from_numpy(bands[2]).float().to(device)  # B3
                red = torch.from_numpy(bands[3]).float().to(device)    # B4
                nir = torch.from_numpy(bands[7]).float().to(device)    # B8
                swir = torch.from_numpy(bands[10]).float().to(device)  # B11
                
                eps = 1e-6
                
                # ✅ CORRECT formulas
                ndvi = (nir - red) / (nir + red + eps)
                ndbi = (swir - nir) / (swir + nir + eps)
                ndwi = (green - nir) / (green + nir + eps)  # Water index
                
                # Filter valid pixels
                mask = (red > 0) & (nir > 0) & (swir > 0)
                ndvi_valid = ndvi[mask]
                ndbi_valid = ndbi[mask]
                ndwi_valid = ndwi[mask]
                
                # 🔥 RATIO FEATURES (CRITICAL!)
                total_pixels = float(torch.numel(ndvi))
                builtup_ratio = float(torch.sum(ndbi > 0).cpu()) / total_pixels
                dense_urban_ratio = float(torch.sum(ndbi > 0.1).cpu()) / total_pixels
                vegetation_ratio = float(torch.sum(ndvi > 0.3).cpu()) / total_pixels
                dense_veg_ratio = float(torch.sum(ndvi > 0.5).cpu()) / total_pixels
                water_ratio = float(torch.sum(ndwi > 0.3).cpu()) / total_pixels
                barren_ratio = float(torch.sum((ndvi < 0.1) & (ndbi < 0.1)).cpu()) / total_pixels
                
                # 🔥 TEXTURE FEATURES
                # Sobel edge detection
                sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=device)
                sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=device)
                
                red_pad = torch.nn.functional.pad(red.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='reflect')
                grad_x = torch.nn.functional.conv2d(red_pad, sobel_x.unsqueeze(0).unsqueeze(0))
                grad_y = torch.nn.functional.conv2d(red_pad, sobel_y.unsqueeze(0).unsqueeze(0))
                edge_magnitude = torch.sqrt(grad_x**2 + grad_y**2).squeeze()
                
                edge_density = float(torch.mean(edge_magnitude).cpu())
                edge_std = float(torch.std(edge_magnitude).cpu())
                
                # Local variance (texture)
                red_mean = torch.nn.functional.avg_pool2d(red.unsqueeze(0).unsqueeze(0), 5, stride=1, padding=2)
                red_var = torch.nn.functional.avg_pool2d((red.unsqueeze(0).unsqueeze(0) - red_mean)**2, 5, stride=1, padding=2)
                texture_mean = float(torch.mean(red_var).cpu())
                
                # 🔥 MULTI-SCALE FEATURES
                h, w = ndvi.shape
                # Center crop (urban core)
                center_h, center_w = h//4, w//4
                ndvi_center = ndvi[center_h:3*center_h, center_w:3*center_w]
                ndbi_center = ndbi[center_h:3*center_h, center_w:3*center_w]
                
                # Quadrants
                ndvi_q1 = ndvi[:h//2, :w//2]
                ndvi_q2 = ndvi[:h//2, w//2:]
                ndvi_q3 = ndvi[h//2:, :w//2]
                ndvi_q4 = ndvi[h//2:, w//2:]
                
                # Heterogeneity (spatial variation)
                ndvi_heterogeneity = float(torch.std(torch.tensor([
                    torch.mean(ndvi_q1).cpu(), torch.mean(ndvi_q2).cpu(),
                    torch.mean(ndvi_q3).cpu(), torch.mean(ndvi_q4).cpu()
                ])).cpu())
                
                return {
                    # Basic stats
                    'ndvi_mean': float(torch.mean(ndvi_valid).cpu()) if len(ndvi_valid) > 0 else 0.0,
                    'ndvi_std': float(torch.std(ndvi_valid).cpu()) if len(ndvi_valid) > 0 else 0.0,
                    'ndvi_p10': float(torch.quantile(ndvi_valid, 0.1).cpu()) if len(ndvi_valid) > 0 else 0.0,
                    'ndvi_p90': float(torch.quantile(ndvi_valid, 0.9).cpu()) if len(ndvi_valid) > 0 else 0.0,
                    'ndbi_mean': float(torch.mean(ndbi_valid).cpu()) if len(ndbi_valid) > 0 else 0.0,
                    'ndbi_std': float(torch.std(ndbi_valid).cpu()) if len(ndbi_valid) > 0 else 0.0,
                    'ndwi_mean': float(torch.mean(ndwi_valid).cpu()) if len(ndwi_valid) > 0 else 0.0,
                    'red_mean': float(torch.mean(red[mask]).cpu()) if torch.sum(mask) > 0 else 0.0,
                    'red_std': float(torch.std(red[mask]).cpu()) if torch.sum(mask) > 0 else 0.0,
                    'nir_mean': float(torch.mean(nir[mask]).cpu()) if torch.sum(mask) > 0 else 0.0,
                    'blue_mean': float(torch.mean(blue[mask]).cpu()) if torch.sum(mask) > 0 else 0.0,
                    'green_mean': float(torch.mean(green[mask]).cpu()) if torch.sum(mask) > 0 else 0.0,
                    'swir_mean': float(torch.mean(swir[mask]).cpu()) if torch.sum(mask) > 0 else 0.0,
                    # 🔥 RATIO FEATURES
                    'builtup_ratio': builtup_ratio,
                    'dense_urban_ratio': dense_urban_ratio,
                    'vegetation_ratio': vegetation_ratio,
                    'dense_veg_ratio': dense_veg_ratio,
                    'water_ratio': water_ratio,
                    'barren_ratio': barren_ratio,
                    # 🔥 TEXTURE FEATURES
                    'edge_density': edge_density,
                    'edge_std': edge_std,
                    'texture_mean': texture_mean,
                    # 🔥 MULTI-SCALE FEATURES
                    'ndvi_center': float(torch.mean(ndvi_center).cpu()) if ndvi_center.numel() > 0 else 0.0,
                    'ndbi_center': float(torch.mean(ndbi_center).cpu()) if ndbi_center.numel() > 0 else 0.0,
                    'ndvi_heterogeneity': ndvi_heterogeneity,
                }
            else:
                # VIIRS single band
                band = torch.from_numpy(bands[0]).float().to(device)
                band_valid = band[band > 0]  # Filter zeros
                return {
                    'brightness_mean': float(torch.mean(band_valid).cpu()) if len(band_valid) > 0 else 0.0,
                    'brightness_std': float(torch.std(band_valid).cpu()) if len(band_valid) > 0 else 0.0,
                    'brightness_p10': float(torch.quantile(band_valid, 0.1).cpu()) if len(band_valid) > 0 else 0.0,
                    'brightness_p90': float(torch.quantile(band_valid, 0.9).cpu()) if len(band_valid) > 0 else 0.0,
                }
    except Exception as e:
        return {}

def build_cache_gpu(df, composite_dir, cache_path):
    """Build cache with GPU acceleration"""
    if Path(cache_path).exists():
        print(f'✅ Cache exists: {cache_path}')
        return pd.read_csv(cache_path)
    
    print(f'🚀 GPU extracting: {cache_path}')
    rows = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc='GPU Processing'):
        s_path = composite_dir / row['sentinel2_tiff_file_name']
        v_path = composite_dir / row['viirs_tiff_file_name']
        
        s_feats = extract_features_gpu(s_path)
        v_feats = extract_features_gpu(v_path)
        
        feats = {f'sentinel_{k}': v for k, v in s_feats.items()}
        feats.update({f'viirs_{k}': v for k, v in v_feats.items()})
        feats['data_id'] = row['data_id']
        
        rows.append(feats)
        
        # Clear GPU cache every 50 images
        if idx % 50 == 0:
            torch.cuda.empty_cache()
    
    result = pd.DataFrame(rows)
    result.to_csv(cache_path, index=False)
    print(f'✅ Saved: {cache_path} ({len(result)} rows)')
    return result

if __name__ == '__main__':
    import time
    start = time.time()
    
    print('\n🔥 ADVANCED FEATURE EXTRACTION')
    print('='*60)
    print('✅ Correct Sentinel-2 bands (B2,B3,B4,B8,B11)')
    print('✅ NDVI, NDBI, NDWI indices')
    print('🔥 Ratio features: builtup, vegetation, water, barren')
    print('🔥 Texture features: edge density, local variance')
    print('🔥 Multi-scale: center crop, quadrants, heterogeneity')
    print('='*60)
    
    train = pd.read_csv(DATA_DIR / 'train_tabular.csv')
    eval_df = pd.read_csv(DATA_DIR / 'evaluation_tabular_no_target.csv')
    
    print(f'\n🚀 Processing {len(train)} train images...')
    train_cache = build_cache_gpu(
        train, 
        DATA_DIR / 'train_composite',
        DATA_DIR / 'sat_features_train.csv'
    )
    
    print(f'\n🚀 Processing {len(eval_df)} eval images...')
    eval_cache = build_cache_gpu(
        eval_df,
        DATA_DIR / 'evaluation_composite', 
        DATA_DIR / 'sat_features_eval.csv'
    )
    
    elapsed = (time.time() - start) / 60
    print(f'\n✅ DONE in {elapsed:.1f} minutes!')
    print(f'Train: {train_cache.shape}, Eval: {eval_cache.shape}')
