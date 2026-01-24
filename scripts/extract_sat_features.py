import os
import pickle
from pathlib import Path
import rasterio
from satellite_features import band_percentiles, indices_from_bands, multi_scale_stats, viirs_contrast
import numpy as np

try:
    from scipy.ndimage import gaussian_filter
except Exception:
    gaussian_filter = None

BASE = Path('d:/Projects/Construction-cost')
DATA_DIR = BASE / 'datasets'
CACHE_DIR = BASE / 'cache' / 'sat_v1'
CACHE_DIR.mkdir(parents=True, exist_ok=True)

CACHE_DIR_V2 = BASE / 'cache' / 'sat_v2'
CACHE_DIR_V2.mkdir(parents=True, exist_ok=True)

def _tile_key(sentinel_fname, viirs_fname):
    s = Path(sentinel_fname).stem if sentinel_fname and not pd_isnan(sentinel_fname) else 'none'
    v = Path(viirs_fname).stem if viirs_fname and not pd_isnan(viirs_fname) else 'none'
    return f'{s}__{v}'

def pd_isnan(x):
    try:
        import pandas as pd
        return pd.isna(x)
    except Exception:
        return x is None

def read_raster(path):
    try:
        with rasterio.open(path) as ds:
            arr = ds.read()
            return arr
    except Exception:
        return None

def bands_from_raster(arr):
    # arr: bands, H, W
    if arr is None:
        return {}
    if arr.ndim == 2:
        # single band
        return {'band1': arr}
    bcount = arr.shape[0]
    bands = {}
    # heuristic mapping: 0=blue,1=green,2=red,3=nir,4=swir1
    if bcount >= 4:
        bands['red'] = arr[2]
        bands['nir'] = arr[3]
        bands['green'] = arr[1]
        if bcount >= 5:
            bands['swir1'] = arr[4]
    else:
        # fallback: name by index
        for i in range(bcount):
            bands[f'band{i+1}'] = arr[i]
    return bands

def extract_features_for_pair(sentinel_fname, viirs_fname, lat=None, lon=None):
    key = _tile_key(sentinel_fname, viirs_fname)
    # include lat/lon in cache key if provided (for true ring stats)
    if lat is not None and lon is not None:
        key = f'{key}_{lat:.6f}_{lon:.6f}'
    cache_file_v1 = CACHE_DIR / f'{key}.pkl'
    cache_file_v2 = CACHE_DIR_V2 / f'{key}.pkl'
    # prefer v2 cache
    if cache_file_v2.exists():
        try:
            with open(cache_file_v2, 'rb') as f:
                return pickle.load(f)
        except Exception:
            pass

    # if v1 exists, load and augment
    features = {}
    if cache_file_v1.exists():
        try:
            with open(cache_file_v1, 'rb') as f:
                features = pickle.load(f)
        except Exception:
            features = {}

    # build full paths
    sentinel_path = DATA_DIR / sentinel_fname if sentinel_fname and not pd_isnan(sentinel_fname) else None
    viirs_path = DATA_DIR / viirs_fname if viirs_fname and not pd_isnan(viirs_fname) else None

    sat_arr = read_raster(sentinel_path) if sentinel_path else None
    viirs_arr = None
    if viirs_path:
        v = read_raster(viirs_path)
        if v is not None:
            # take first band
            viirs_arr = v[0] if v.ndim == 3 else v

    bands = bands_from_raster(sat_arr)

    # compute base (v1) features if not already present
    if not features:
        if bands:
            features.update(band_percentiles(bands))
            features.update(indices_from_bands(bands))
            features.update(multi_scale_stats(bands))
        else:
            features['no_sat_data'] = 1

    if viirs_arr is not None:
        features.update(viirs_contrast(viirs_arr))
    else:
        features['no_viirs'] = 1

    # ----------------------
    # V2: Concentric context + texture
    # ----------------------
    try:
        pixel_size = None
        if sentinel_path:
            try:
                with rasterio.open(sentinel_path) as ds:
                    px = abs(ds.transform.a)
                    py = abs(ds.transform.e)
                    pixel_size = float(max(px, py))
            except Exception:
                pixel_size = None
        if pixel_size is None or pixel_size == 0:
            pixel_size = 10.0

        def integral_img(a):
            a = np.nan_to_num(a, nan=0.0)
            return a.cumsum(axis=0).cumsum(axis=1)

        def box_sum(ii, y1, x1, y2, x2):
            y1 = max(0, y1); x1 = max(0, x1)
            y2 = min(ii.shape[0]-1, y2); x2 = min(ii.shape[1]-1, x2)
            A = ii[y2, x2]
            B = ii[y1-1, x2] if y1-1 >= 0 else 0
            C = ii[y2, x1-1] if x1-1 >= 0 else 0
            D = ii[y1-1, x1-1] if (y1-1>=0 and x1-1>=0) else 0
            return A - B - C + D

        if bands:
            ndvi = None
            ndbi = None
            if 'nir' in bands and 'red' in bands:
                nir = bands['nir'].astype(float)
                red = bands['red'].astype(float)
                ndvi = (nir - red) / (np.maximum(1e-6, (nir + red)))
            if 'swir1' in bands and 'nir' in bands:
                swir1 = bands['swir1'].astype(float)
                ndbi = (swir1 - bands['nir'].astype(float)) / (np.maximum(1e-6, (swir1 + bands['nir'].astype(float))))

            H = next(iter(bands.values())).shape[0]
            W = next(iter(bands.values())).shape[1]
            cy = H // 2; cx = W // 2

            radii_m = [0, 1000, 3000, 5000]
            radii_px = [max(1, int(r / pixel_size)) for r in radii_m]

            ii_ndvi = integral_img(ndvi) if ndvi is not None else None
            ii_ndbi = integral_img(ndbi) if ndbi is not None else None

            def mean_in_radius(ii, r_px):
                if ii is None:
                    return np.nan
                y1 = cy - r_px; x1 = cx - r_px; y2 = cy + r_px; x2 = cx + r_px
                s = box_sum(ii, y1, x1, y2, x2)
                area = (y2 - y1 + 1) * (x2 - x1 + 1)
                return float(s / max(1, area))

            ndvi_means = []
            ndbi_means = []
            for r in radii_px:
                ndvi_means.append(mean_in_radius(ii_ndvi, r))
                ndbi_means.append(mean_in_radius(ii_ndbi, r))

            for i, r in enumerate(['center','r1','r2','r3']):
                features[f'ndvi_{r}_mean'] = ndvi_means[i]
                features[f'ndbi_{r}_mean'] = ndbi_means[i]

            features['ndvi_r1_center_diff'] = (ndvi_means[1] - ndvi_means[0]) if ndvi_means[0] is not None else np.nan
            features['ndvi_r2_r1_diff'] = (ndvi_means[2] - ndvi_means[1]) if ndvi_means[1] is not None else np.nan
            features['ndvi_r3_r2_diff'] = (ndvi_means[3] - ndvi_means[2]) if ndvi_means[2] is not None else np.nan

            for i, r_px in enumerate(radii_px):
                if ndbi is None:
                    features[f'builtup_prop_{i}'] = np.nan
                else:
                    y1 = cy - r_px; x1 = cx - r_px; y2 = cy + r_px; x2 = cx + r_px
                    sub = ndbi[max(0,y1):min(H,y2+1), max(0,x1):min(W,x2+1)]
                    features[f'builtup_prop_{i}'] = float(np.sum(sub > 0) / max(1, np.sum(~np.isnan(sub))))

            try:
                if ndvi is not None:
                    features.update({f'ndvi_{k}': v for k, v in multi_scale_stats({'ndvi': ndvi}).items()})
                if ndbi is not None:
                    features.update({f'ndbi_{k}': v for k, v in multi_scale_stats({'ndbi': ndbi}).items()})
            except Exception:
                pass

        # ----------------------
        # V3: TRUE RINGS via geotransform (location-based spatial context)
        # ----------------------
        try:
            if lat is not None and lon is not None and sentinel_path:
                with rasterio.open(sentinel_path) as ds:
                    # map lat/lon to pixel coordinates
                    transform = ds.transform
                    # rasterio uses affine transform: (x, y) = (col, row) in pixel space
                    # inverse transform: (col, row) = ~transform * (lon, lat)
                    col, row = ~transform * (lon, lat)
                    col, row = int(round(col)), int(round(row))
                    H, W = ds.shape
                    
                    # clamp to bounds
                    col = max(0, min(W-1, col))
                    row = max(0, min(H-1, row))
                    
                    # define rings by pixel distance from the location center
                    def dist_map(h, w, center_r, center_c):
                        yy, xx = np.ogrid[:h, :w]
                        return np.sqrt((yy - center_r)**2 + (xx - center_c)**2)
                    
                    dist = dist_map(H, W, row, col)
                    
                    # ring definitions (in pixel distance)
                    ring_defs = [
                        ('center', 0, 2),
                        ('ring1', 2, 8),
                        ('ring2', 8, 25),
                        ('ring3', 25, 80)
                    ]
                    
                    if ndvi is not None:
                        ndvi_ring_stats = {}
                        for ring_name, r_min, r_max in ring_defs:
                            mask = (dist > r_min) & (dist <= r_max)
                            if np.any(mask):
                                vals = ndvi[mask]
                                vals = vals[~np.isnan(vals)]
                                if len(vals) > 0:
                                    ndvi_ring_stats[ring_name] = {
                                        'mean': float(np.mean(vals)),
                                        'p90': float(np.percentile(vals, 90)) if len(vals) >= 10 else float(np.mean(vals))
                                    }
                                else:
                                    ndvi_ring_stats[ring_name] = {'mean': np.nan, 'p90': np.nan}
                            else:
                                ndvi_ring_stats[ring_name] = {'mean': np.nan, 'p90': np.nan}
                        
                        # save ring stats
                        for ring_name in [k[0] for k in ring_defs]:
                            if ring_name in ndvi_ring_stats:
                                features[f'ndvi_ring_{ring_name}_mean'] = ndvi_ring_stats[ring_name]['mean']
                                features[f'ndvi_ring_{ring_name}_p90'] = ndvi_ring_stats[ring_name]['p90']
                        
                        # compute deltas
                        if 'center' in ndvi_ring_stats and 'ring1' in ndvi_ring_stats:
                            c_mean = ndvi_ring_stats['center']['mean']
                            r1_mean = ndvi_ring_stats['ring1']['mean']
                            if c_mean is not None and not np.isnan(c_mean) and r1_mean is not None and not np.isnan(r1_mean):
                                features['ndvi_ring1_delta_from_center'] = r1_mean - c_mean
                        
                        if 'ring1' in ndvi_ring_stats and 'ring2' in ndvi_ring_stats:
                            r1_mean = ndvi_ring_stats['ring1']['mean']
                            r2_mean = ndvi_ring_stats['ring2']['mean']
                            if r1_mean is not None and not np.isnan(r1_mean) and r2_mean is not None and not np.isnan(r2_mean):
                                features['ndvi_ring2_delta_from_ring1'] = r2_mean - r1_mean
                        
                        if 'ring2' in ndvi_ring_stats and 'ring3' in ndvi_ring_stats:
                            r2_mean = ndvi_ring_stats['ring2']['mean']
                            r3_mean = ndvi_ring_stats['ring3']['mean']
                            if r2_mean is not None and not np.isnan(r2_mean) and r3_mean is not None and not np.isnan(r3_mean):
                                features['ndvi_ring3_delta_from_ring2'] = r3_mean - r2_mean
                    
                    if ndbi is not None:
                        ndbi_ring_stats = {}
                        for ring_name, r_min, r_max in ring_defs:
                            mask = (dist > r_min) & (dist <= r_max)
                            if np.any(mask):
                                vals = ndbi[mask]
                                vals = vals[~np.isnan(vals)]
                                if len(vals) > 0:
                                    ndbi_ring_stats[ring_name] = {
                                        'mean': float(np.mean(vals)),
                                        'p90': float(np.percentile(vals, 90)) if len(vals) >= 10 else float(np.mean(vals))
                                    }
                                else:
                                    ndbi_ring_stats[ring_name] = {'mean': np.nan, 'p90': np.nan}
                            else:
                                ndbi_ring_stats[ring_name] = {'mean': np.nan, 'p90': np.nan}
                        
                        for ring_name in [k[0] for k in ring_defs]:
                            if ring_name in ndbi_ring_stats:
                                features[f'ndbi_ring_{ring_name}_mean'] = ndbi_ring_stats[ring_name]['mean']
                                features[f'ndbi_ring_{ring_name}_p90'] = ndbi_ring_stats[ring_name]['p90']
                    
                    if viirs_arr is not None:
                        vi = viirs_arr.astype(float)
                        viirs_ring_stats = {}
                        for ring_name, r_min, r_max in ring_defs:
                            mask = (dist > r_min) & (dist <= r_max)
                            if np.any(mask):
                                vals = vi[mask]
                                vals = vals[~np.isnan(vals)]
                                if len(vals) > 0:
                                    viirs_ring_stats[ring_name] = {
                                        'mean': float(np.mean(vals)),
                                        'p90': float(np.percentile(vals, 90)) if len(vals) >= 10 else float(np.mean(vals))
                                    }
                                else:
                                    viirs_ring_stats[ring_name] = {'mean': np.nan, 'p90': np.nan}
                            else:
                                viirs_ring_stats[ring_name] = {'mean': np.nan, 'p90': np.nan}
                        
                        for ring_name in [k[0] for k in ring_defs]:
                            if ring_name in viirs_ring_stats:
                                features[f'viirs_ring_{ring_name}_mean'] = viirs_ring_stats[ring_name]['mean']
                                features[f'viirs_ring_{ring_name}_p90'] = viirs_ring_stats[ring_name]['p90']
        except Exception as e:
            # ring computation failed, continue with other features
            pass
        
        # VIIRS multi-scale context via gaussian blur / pooling
        try:
            if viirs_arr is not None:
                vi = viirs_arr.astype(float)
                # scales in meters
                scales_m = [0, 1000, 3000, 5000, 10000]
                tile_stats = {}
                for r in scales_m:
                    if r == 0:
                        arr = vi
                    else:
                        if gaussian_filter is not None:
                            sigma = max(1.0, (r / pixel_size) / 3.0)
                            arr = gaussian_filter(vi, sigma=sigma)
                        else:
                            # fallback: simple uniform downsample by resizing
                            factor = max(1, int((r / pixel_size) / 4))
                            if factor <= 1:
                                arr = vi
                            else:
                                # block mean pooling
                                h, w = vi.shape
                                ph = (factor - (h % factor)) % factor
                                pw = (factor - (w % factor)) % factor
                                vpad = np.pad(vi, ((0, ph), (0, pw)), mode='constant', constant_values=np.nan)
                                vrs = vpad.reshape(vpad.shape[0]//factor, factor, vpad.shape[1]//factor, factor)
                                arr = np.nanmean(vrs, axis=(1,3))
                                # upsample back to tile size
                                arr = np.repeat(np.repeat(arr, factor, axis=0), factor, axis=1)[:h, :w]
                    m = float(np.nanmean(arr)) if arr.size else np.nan
                    p90 = float(np.nanpercentile(arr, 90)) if arr.size else np.nan
                    p99 = float(np.nanpercentile(arr, 99)) if arr.size else np.nan
                    tile_stats[r] = (m, p90, p99)
                    features[f'viirs_mean_{r}m'] = m
                    features[f'viirs_p90_{r}m'] = p90
                    features[f'viirs_p99_{r}m'] = p99
                # deltas vs tile
                base_m, base_p90, base_p99 = tile_stats[0]
                for r in scales_m[1:]:
                    mr, p90r, p99r = tile_stats[r]
                    features[f'viirs_mean_{r}m_minus_tile'] = (mr - base_m) if (mr is not None and base_m is not None) else np.nan
                    features[f'viirs_p90_{r}m_minus_tile'] = (p90r - base_p90) if (p90r is not None and base_p90 is not None) else np.nan
                    features[f'viirs_p99_{r}m_minus_tile'] = (p99r - base_p99) if (p99r is not None and base_p99 is not None) else np.nan
        except Exception:
            pass

    except Exception:
        pass

    # save v1 cache if not present
    try:
        if not cache_file_v1.exists():
            with open(cache_file_v1, 'wb') as f:
                pickle.dump(features, f)
    except Exception:
        pass

    # save v2 cache
    try:
        with open(cache_file_v2, 'wb') as f:
            pickle.dump(features, f)
    except Exception:
        pass

    return features

if __name__ == '__main__':
    # quick self-test over a couple of files in DATA_DIR
    import glob
    samples = list((DATA_DIR).glob('**/*.tif'))[:4]
    for s in samples:
        sf = s.name
        feats = extract_features_for_pair(sf, None)
        print(sf, '->', list(feats.keys())[:6])
