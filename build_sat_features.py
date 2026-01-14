"""
One-time satellite feature extraction with caching
Run this ONCE, then use cached CSV files for all experiments
"""
import numpy as np
import pandas as pd
from pathlib import Path
import rasterio
from scipy.ndimage import generic_filter
from tqdm.auto import tqdm

BASE = Path('d:/Projects/Construction-cost')
DATA_DIR = BASE / 'datasets'

def extract_features(tif_path):
    """Extract satellite features efficiently"""
    try:
        with rasterio.open(tif_path) as src:
            bands = src.read()
            
            if bands.shape[0] >= 4:
                blue, green, red, nir = bands[0], bands[1], bands[2], bands[3]
                
                # NDVI
                ndvi = np.where((nir + red) > 0, (nir - red) / (nir + red), 0)
                
                # NDBI
                ndbi = np.where((red + nir) > 0, (red - nir) / (red + nir), 0)
                
                # Texture
                red_var = generic_filter(red.astype(float), np.var, size=5)
                
                return {
                    'ndvi_mean': np.nanmean(ndvi),
                    'ndvi_std': np.nanstd(ndvi),
                    'ndvi_p10': np.nanpercentile(ndvi, 10),
                    'ndvi_p90': np.nanpercentile(ndvi, 90),
                    'ndbi_mean': np.nanmean(ndbi),
                    'ndbi_std': np.nanstd(ndbi),
                    'red_mean': np.nanmean(red),
                    'red_std': np.nanstd(red),
                    'red_texture': np.nanmean(red_var),
                    'nir_mean': np.nanmean(nir),
                    'blue_mean': np.nanmean(blue),
                    'green_mean': np.nanmean(green),
                }
            else:
                band = bands[0]
                return {
                    'brightness_mean': np.nanmean(band),
                    'brightness_std': np.nanstd(band),
                    'brightness_p10': np.nanpercentile(band, 10),
                    'brightness_p90': np.nanpercentile(band, 90),
                }
    except:
        return {}

def build_cache(df, composite_dir, cache_path, prefix=''):
    """Build satellite feature cache"""
    if Path(cache_path).exists():
        print(f'✅ Cache exists: {cache_path}')
        return pd.read_csv(cache_path)
    
    print(f'🔄 Building cache: {cache_path}')
    rows = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        s_path = composite_dir / row['sentinel2_tiff_file_name']
        v_path = composite_dir / row['viirs_tiff_file_name']
        
        s_feats = extract_features(s_path)
        v_feats = extract_features(v_path)
        
        feats = {f'sentinel_{k}': v for k, v in s_feats.items()}
        feats.update({f'viirs_{k}': v for k, v in v_feats.items()})
        feats['data_id'] = row['data_id']
        
        rows.append(feats)
    
    result = pd.DataFrame(rows)
    result.to_csv(cache_path, index=False)
    print(f'✅ Saved: {cache_path} ({len(result)} rows)')
    return result

if __name__ == '__main__':
    print('='*60)
    print('⚠️  CPU FEATURE EXTRACTION (Slow but works)')
    print('='*60)
    print('Expected time: ~4 hours for 1024 images')
    print('This runs ONCE. Then training will be fast.')
    print('='*60)
    
    # Load data
    train = pd.read_csv(DATA_DIR / 'train_tabular.csv')
    eval_df = pd.read_csv(DATA_DIR / 'evaluation_tabular_no_target.csv')
    
    # Build caches
    train_cache = build_cache(
        train, 
        DATA_DIR / 'train_composite',
        DATA_DIR / 'sat_features_train.csv'
    )
    
    eval_cache = build_cache(
        eval_df,
        DATA_DIR / 'evaluation_composite', 
        DATA_DIR / 'sat_features_eval.csv'
    )
    
    print('\n✅ Done! Use these cached files in your modeling notebook.')
    print(f'Train features: {train_cache.shape}')
    print(f'Eval features: {eval_cache.shape}')
