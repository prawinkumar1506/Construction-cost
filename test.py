import rasterio

p = r"d:\Projects\Construction-cost\datasets\train_composite\sentinel_2_01000_hokkaido_2019-Q3.tif"

with rasterio.open(p) as src:
    print("Bands:", src.count)
    print("Shape:", src.read().shape)
