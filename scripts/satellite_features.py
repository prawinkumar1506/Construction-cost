import numpy as np

def pct_stats(arr, percentiles=(1,5,10,25,50,75,90,95,99)):
    arr = np.asarray(arr).ravel()
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return {f'p{p}': np.nan for p in percentiles}
    qs = np.percentile(arr, percentiles)
    return {f'p{int(p)}': float(q) for p, q in zip(percentiles, qs)}

def band_percentiles(bands, percentiles=(1,5,10,25,50,75,90,95,99)):
    """Compute percentiles per band.

    bands: dict of name -> 2D ndarray
    returns: dict of feature_name -> value
    """
    out = {}
    for name, arr in bands.items():
        stats = pct_stats(arr, percentiles)
        for k, v in stats.items():
            out[f'{name}_{k}'] = v
    return out

def safe_ratio(a, b, eps=1e-6):
    a = np.asarray(a).astype(float)
    b = np.asarray(b).astype(float)
    denom = b.copy()
    denom[np.abs(denom) < eps] = eps
    return a / denom

def indices_from_bands(bands):
    """Compute common indices when bands exist.

    Expected band keys: 'red', 'nir', 'green', 'swir1'
    """
    out = {}
    if 'nir' in bands and 'red' in bands:
        ndvi = safe_ratio(bands['nir'] - bands['red'], bands['nir'] + bands['red'])
        out.update({f'ndvi_{k}': v for k, v in pct_stats(ndvi).items()})
    if 'green' in bands and 'nir' in bands:
        ndwi = safe_ratio(bands['green'] - bands['nir'], bands['green'] + bands['nir'])
        out.update({f'ndwi_{k}': v for k, v in pct_stats(ndwi).items()})
    if 'swir1' in bands and 'nir' in bands:
        ndbi = safe_ratio(bands['swir1'] - bands['nir'], bands['swir1'] + bands['nir'])
        out.update({f'ndbi_{k}': v for k, v in pct_stats(ndbi).items()})
    return out

def block_reduce_stats(arr, block_size):
    """Compute block-wise mean and std by simple reshaping (pads if necessary)."""
    h, w = arr.shape
    bh, bw = block_size
    # pad
    ph = (-h) % bh
    pw = (-w) % bw
    if ph != 0 or pw != 0:
        arr = np.pad(arr, ((0, ph), (0, pw)), mode='constant', constant_values=np.nan)
    h2, w2 = arr.shape
    arr = arr.reshape(h2//bh, bh, w2//bw, bw)
    # move blocks to first axes
    arr_blocks = arr.swapaxes(1,2).reshape(-1, bh, bw)
    means = np.nanmean(arr_blocks, axis=(1,2))
    stds = np.nanstd(arr_blocks, axis=(1,2))
    return means, stds

def multi_scale_stats(bands, scales=(8,16,32)):
    out = {}
    for name, arr in bands.items():
        for s in scales:
            means, stds = block_reduce_stats(arr, (s, s))
            out[f'{name}_ms{s}_mean_p50'] = float(np.nanmedian(means))
            out[f'{name}_ms{s}_std_p50'] = float(np.nanmedian(stds))
    return out

def viirs_contrast(viirs_arr):
    viirs = np.asarray(viirs_arr)
    out = {}
    out['viirs_mean'] = float(np.nanmean(viirs))
    out['viirs_std'] = float(np.nanstd(viirs))
    out.update({f'viirs_{k}': v for k, v in pct_stats(viirs).items()})
    # proportion of bright pixels
    thresh = np.nanpercentile(viirs, 95) if np.any(~np.isnan(viirs)) else np.nan
    out['viirs_prop95'] = float(np.sum(viirs >= thresh) / np.maximum(1, np.sum(~np.isnan(viirs))))
    return out


if __name__ == '__main__':
    # simple demo
    H, W = 256, 256
    bands = {
        'red': np.random.rand(H, W) * 0.3 + 0.1,
        'nir': np.random.rand(H, W) * 0.5 + 0.2,
        'green': np.random.rand(H, W) * 0.2 + 0.05,
        'swir1': np.random.rand(H, W) * 0.4 + 0.1,
    }
    viirs = np.random.rand(H, W) * 5.0
    print('Band percentiles sample:')
    print(band_percentiles(bands))
    print('Index stats sample:')
    print(indices_from_bands(bands))
    print('Multi-scale sample:')
    print(multi_scale_stats({'red': bands['red']}))
    print('VIIRS sample:')
    print(viirs_contrast(viirs))
