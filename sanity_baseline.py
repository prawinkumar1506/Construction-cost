# import pandas as pd
# import numpy as np
# from sklearn.model_selection import GroupKFold
# from sklearn.metrics import mean_absolute_error

# train = pd.read_csv("datasets/train_tabular.csv")

# # Baseline: predict mean per location
# gkf = GroupKFold(5)
# groups = train["geolocation_name"]

# oof = np.zeros(len(train))

# for tr_idx, va_idx in gkf.split(train, train["construction_cost_per_m2_usd"], groups):
#     tr = train.iloc[tr_idx]
#     va = train.iloc[va_idx]

#     means = tr.groupby("geolocation_name")["construction_cost_per_m2_usd"].mean()
#     pred = va["geolocation_name"].map(means)

#     global_mean = tr["construction_cost_per_m2_usd"].mean()
#     pred = pred.fillna(global_mean)

#     oof[va_idx] = pred

# mae = mean_absolute_error(train["construction_cost_per_m2_usd"], oof)
# print("Location mean baseline MAE:", mae)


# import pandas as pd

# train = pd.read_csv("datasets/train_tabular.csv")

# print("Rows:", len(train))
# print("Unique locations:", train["geolocation_name"].nunique())
# print("Avg repeats:", len(train) / train["geolocation_name"].nunique())
import pandas as pd
import numpy as np

train = pd.read_csv("datasets/train_tabular.csv")

# How much does cost vary inside a location?
loc_stats = train.groupby("geolocation_name")["construction_cost_per_m2_usd"].agg(["mean","std","min","max","count"])

print(loc_stats.describe())

# Average intra-location std
print("Avg intra-location std:", loc_stats["std"].mean())

# Average range
print("Avg intra-location range:", (loc_stats["max"] - loc_stats["min"]).mean())
