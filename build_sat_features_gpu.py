"""
🚀 GPU-ACCELERATED Pipeline for RTX Ada 500
Feature extraction + Training in MINUTES instead of HOURS
"""
import numpy as np
import pandas as pd
from pathlib import Path
import rasterio
from tqdm.auto import tqdm
import cupy as cp  # GPU arrays
from cupyx.scipy.ndimage import generic_filter as gpu_generic_filter

BASE = Path('d:/Projects/Construction-cost')
DATA_DIR = BASE / 'datasets'

def extract_features_gpu(tif_path):
    """GPU-accelerated feature extraction"""
    try:
        with rasterio.open(tif_path) as src:
            bands = src.read()
            
            if bands.shape[0] >= 4:
                # Move to GPU
                blue = cp.asarray(bands[0], dtype=cp.float32)
                green = cp.asarray(bands[1], dtype=cp.float32)
                red = cp.asarray(bands[2], dtype=cp.float32)
                nir = cp.asarray(bands[3], dtype=cp.float32)
                
                # GPU computations
                ndvi = cp.where((nir + red) > 0, (nir - red) / (nir + red), 0)
                ndbi = cp.where((red + nir) > 0, (red - nir) / (red + nir), 0)
                
                # Texture on GPU
                red_var = gpu_generic_filter(red, cp.var, size=5)
                
                # Move back to CPU for storage
                return {
                    'ndvi_mean': float(cp.nanmean(ndvi).get()),
                    'ndvi_std': float(cp.nanstd(ndvi).get()),
                    'ndvi_p10': float(cp.nanpercentile(ndvi, 10).get()),
                    'ndvi_p90': float(cp.nanpercentile(ndvi, 90).get()),
                    'ndbi_mean': float(cp.nanmean(ndbi).get()),
                    'ndbi_std': float(cp.nanstd(ndbi).get()),
                    'red_mean': float(cp.nanmean(red).get()),
                    'red_std': float(cp.nanstd(red).get()),
                    'red_texture': float(cp.nanmean(red_var).get()),
                    'nir_mean': float(cp.nanmean(nir).get()),
                    'blue_mean': float(cp.nanmean(blue).get()),
                    'green_mean': float(cp.nanmean(green).get()),
                }
            else:
                band = cp.asarray(bands[0], dtype=cp.float32)
                return {
                    'brightness_mean': float(cp.nanmean(band).get()),
                    'brightness_std': float(cp.nanstd(band).get()),
                    'brightness_p10': float(cp.nanpercentile(band, 10).get()),
                    'brightness_p90': float(cp.nanpercentile(band, 90).get()),
                }
    except:
        return {}

def build_cache_gpu(df, composite_dir, cache_path):
    """GPU-accelerated cache building"""
    if Path(cache_path).exists():
        print(f'✅ Cache exists: {cache_path}')
        return pd.read_csv(cache_path)
    
    print(f'🚀 GPU building: {cache_path}')
    rows = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        s_path = composite_dir / row['sentinel2_tiff_file_name']
        v_path = composite_dir / row['viirs_tiff_file_name']
        
        s_feats = extract_features_gpu(s_path)
        v_feats = extract_features_gpu(v_path)
        
        feats = {f'sentinel_{k}': v for k, v in s_feats.items()}
        feats.update({f'viirs_{k}': v for k, v in v_feats.items()})
        feats['data_id'] = row['data_id']
        
        rows.append(feats)
    
    result = pd.DataFrame(rows)
    result.to_csv(cache_path, index=False)
    print(f'✅ Saved: {cache_path}')
    return result

if __name__ == '__main__':
    print('🚀 Using GPU:', cp.cuda.Device().compute_capability)
    
    train = pd.read_csv(DATA_DIR / 'train_tabular.csv')
    eval_df = pd.read_csv(DATA_DIR / 'evaluation_tabular_no_target.csv')
    
    train_cache = build_cache_gpu(
        train, 
        DATA_DIR / 'train_composite',
        DATA_DIR / 'sat_features_train.csv'
    )
    
    eval_cache = build_cache_gpu(
        eval_df,
        DATA_DIR / 'evaluation_composite', 
        DATA_DIR / 'sat_features_eval.csv'
    )
    
    print('\n🚀 GPU extraction complete!')
    print(f'Train: {train_cache.shape}, Eval: {eval_cache.shape}')
