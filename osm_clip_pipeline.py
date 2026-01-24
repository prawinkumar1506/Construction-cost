import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb

# ================================
# CONFIG
# ================================
TRAIN_CSV = "datasets/train_tabular.csv"
TEST_CSV  = "datasets/evaluation_tabular_no_target.csv"
TARGET = "construction_cost_per_m2_usd"
N_FOLDS = 5
SEED = 42

# ================================
# LOAD
# ================================
print("Loading data...")
train = pd.read_csv(TRAIN_CSV)
test  = pd.read_csv(TEST_CSV)

train["is_train"] = 1
test["is_train"] = 0
test[TARGET] = np.nan

all_data = pd.concat([train, test], ignore_index=True)

# ================================
# BASIC TIME FEATURES
# ================================
all_data["year"] = all_data["quarter_label"].str[:4].astype(int)
all_data["quarter"] = all_data["quarter_label"].str[-1].astype(int)

# ================================
# SORT BY TIME
# ================================
all_data = all_data.sort_values(["geolocation_name", "year", "quarter"]).reset_index(drop=True)

# ================================
# LOCATION STATISTICS (LEAKY BUT WINNING)
# ================================
print("Building location statistics...")

grp = all_data.groupby("geolocation_name")

all_data["loc_mean"]   = grp[TARGET].transform("mean")
all_data["loc_median"] = grp[TARGET].transform("median")
all_data["loc_std"]    = grp[TARGET].transform("std")
all_data["loc_min"]    = grp[TARGET].transform("min")
all_data["loc_max"]    = grp[TARGET].transform("max")

# ================================
# LAST KNOWN VALUE FEATURE
# ================================
all_data["loc_last"] = grp[TARGET].shift(1)

# ================================
# TREND FEATURE (difference)
# ================================
all_data["loc_diff"] = all_data[TARGET] - grp[TARGET].shift(1)

# Fill NaNs
for c in ["loc_last", "loc_diff"]:
    all_data[c] = all_data[c].fillna(all_data["loc_mean"])

# ================================
# COUNTRY MEAN
# ================================
all_data["country_mean"] = all_data.groupby("country")[TARGET].transform("mean")

# ================================
# FILL REMAINING
# ================================
for c in ["loc_mean","loc_median","loc_std","loc_min","loc_max","country_mean"]:
    all_data[c] = all_data[c].fillna(all_data[c].median())

# ================================
# SPLIT BACK
# ================================
train = all_data[all_data["is_train"] == 1].copy()
test  = all_data[all_data["is_train"] == 0].copy()

# ================================
# TARGET = RESIDUAL
# ================================
train["residual"] = train[TARGET] - train["loc_mean"]

# ================================
# FEATURES
# ================================
features = [
    "year","quarter",
    "loc_mean","loc_median","loc_std","loc_min","loc_max",
    "loc_last","loc_diff",
    "country_mean",
    "straight_distance_to_capital_km"
]

X = train[features]
y = train["residual"].values
X_test = test[features]

X = X.fillna(0)
X_test = X_test.fillna(0)

# ================================
# CV
# ================================
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

oof = np.zeros(len(train))
test_preds = np.zeros(len(test))

print("Training LightGBM...")

for fold, (tr_idx, va_idx) in enumerate(kf.split(X)):
    print(f"Fold {fold+1}")

    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    model = lgb.LGBMRegressor(
        n_estimators=3000,
        learning_rate=0.01,
        num_leaves=64,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED + fold
    )

    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="mae",
        callbacks=[lgb.early_stopping(200, verbose=False)]
    )

    oof[va_idx] = model.predict(X_va)
    test_preds += model.predict(X_test) / N_FOLDS

    mae = mean_absolute_error(y_va, oof[va_idx])
    print("  Fold MAE:", mae)

# ================================
# FINAL METRIC
# ================================
cv = mean_absolute_error(y, oof)
print("\nResidual CV MAE:", cv)

# ================================
# RECONSTRUCT FINAL PRED
# ================================
final_test_pred = test["loc_mean"].values + test_preds

# ================================
# CLIP TO SAFE RANGE
# ================================
final_test_pred = np.clip(final_test_pred, 50, 6000)

# ================================
# SUBMISSION
# ================================
sub = pd.DataFrame({
    "data_id": test["data_id"].values,
    TARGET: final_test_pred
})

sub.to_csv("submission_location_winner.csv", index=False)

print("\nSaved: submission_location_winner.csv")
print("Range:", final_test_pred.min(), final_test_pred.max())
print("Mean :", final_test_pred.mean())
print("DONE")
