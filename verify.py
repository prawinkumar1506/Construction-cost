# # # import pandas as pd

# # # train = pd.read_csv("datasets/train_tabular.csv")
# # # print(train[['data_id','geolocation_name','quarter_label']].head(10))


# # # import os
# # # files = os.listdir("cache/clip_feats")
# # # print(files[:20])
# # import os
# # import numpy as np

# # DIR = "cache/clip_feats"

# # files = os.listdir(DIR)

# # print("Total CLIP files:", len(files))
# # print("First 10 files:", files[:10])

# # # Check shape
# # f = np.load(os.path.join(DIR, files[0]))
# # print("One embedding shape:", f.shape)

# # # Check all shapes
# # shapes = set()
# # for fn in files:
# #     arr = np.load(os.path.join(DIR, fn))
# #     shapes.add(arr.shape)

# # print("Unique shapes:", shapes)


# import pandas as pd
# print(pd.read_csv("datasets/train_tabular.csv").columns.tolist())
import pandas as pd

a = pd.read_csv("submission_lb_matching.csv")
b = pd.read_csv("submission_ultimate_location_time.csv")
c = pd.read_csv("submission_location_clip.csv")

TARGET = "construction_cost_per_m2_usd"

print("LB matching range:", a[TARGET].min(), a[TARGET].max(), a[TARGET].mean())
print("Ultimate range   :", b[TARGET].min(), b[TARGET].max(), b[TARGET].mean())
print("CLIP range       :", c[TARGET].min(), c[TARGET].max(), c[TARGET].mean())
