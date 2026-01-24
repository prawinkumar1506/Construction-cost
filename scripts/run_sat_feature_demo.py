from satellite_features import band_percentiles, indices_from_bands, multi_scale_stats, viirs_contrast
import numpy as np

def main():
    H, W = 256, 256
    bands = {
        'red': np.random.rand(H, W) * 0.3 + 0.1,
        'nir': np.random.rand(H, W) * 0.5 + 0.2,
        'green': np.random.rand(H, W) * 0.2 + 0.05,
        'swir1': np.random.rand(H, W) * 0.4 + 0.1,
    }
    viirs = np.random.rand(H, W) * 5.0
    print('--- Band percentiles ---')
    print(band_percentiles(bands))
    print('\n--- Indices ---')
    print(indices_from_bands(bands))
    print('\n--- Multi-scale ---')
    print(multi_scale_stats({'red': bands['red']}))
    print('\n--- VIIRS ---')
    print(viirs_contrast(viirs))

if __name__ == '__main__':
    main()
