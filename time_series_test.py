import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error

train = pd.read_csv("datasets/train_tabular.csv")

# time index
train["year"] = train["quarter_label"].str[:4].astype(int)
train["q"] = train["quarter_label"].str[-1].astype(int)
train["time_idx"] = train["year"] * 4 + train["q"]

# sort
train = train.sort_values(["geolocation_name","time_idx"]).reset_index(drop=True)

gkf = GroupKFold(5)
groups = train["geolocation_name"]

oof = np.zeros(len(train))

for tr_idx, va_idx in gkf.split(train, train["construction_cost_per_m2_usd"], groups):
    tr = train.iloc[tr_idx]
    va = train.iloc[va_idx]

    # for each location, take last known value from training fold
    tr_sorted = tr.sort_values(["geolocation_name","time_idx"])
    last_map = tr_sorted.groupby("geolocation_name")["construction_cost_per_m2_usd"].last()

    pred = va["geolocation_name"].map(last_map)

    # fallback to global mean
    global_mean = tr["construction_cost_per_m2_usd"].mean()
    pred = pred.fillna(global_mean)

    oof[va_idx] = pred

mae = mean_absolute_error(train["construction_cost_per_m2_usd"], oof)
print("Last-value baseline MAE:", mae)
