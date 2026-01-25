import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

# =========================
# CONFIG
# =========================
TARGET = "construction_cost_per_m2_usd"
TRAIN_CSV = "datasets/train_tabular.csv"
TEST_CSV  = "datasets/evaluation_tabular_no_target.csv"
N_FOLDS = 5
SEED = 42

# =========================
# LOAD
# =========================
print("Loading data...")
train = pd.read_csv(TRAIN_CSV)
test  = pd.read_csv(TEST_CSV)

# =========================
# TIME FEATURES
# =========================
def add_time(df):
    df = df.copy()
    df["year"] = df["quarter_label"].str[:4].astype(int)
    df["q"] = df["quarter_label"].str[-1].astype(int)
    df["time_idx"] = df["year"] * 4 + df["q"]
    return df

train = add_time(train)
test  = add_time(test)

# =========================
# COUNTRY FLAG
# =========================
train["is_japan"] = (train["country"] == "Japan").astype(int)
test["is_japan"]  = (test["country"] == "Japan").astype(int)

# =========================
# BUILD LOCATION HISTORY (INTENTIONALLY FULL-TRAIN LEAKY)
# =========================
print("Building location history features...")

train_sorted = train.sort_values(["geolocation_name","time_idx"])

grp = train_sorted.groupby("geolocation_name")

loc_mean = grp[TARGET].mean()
loc_std  = grp[TARGET].std()
loc_min  = grp[TARGET].min()
loc_max  = grp[TARGET].max()
loc_first = grp[TARGET].first()
loc_last  = grp[TARGET].last()
t_first = grp["time_idx"].first()
t_last  = grp["time_idx"].last()

# slope = change per quarter
loc_slope = (loc_last - loc_first) / (t_last - t_first).replace(0, 1)

# map to train
train["loc_mean"]  = train["geolocation_name"].map(loc_mean)
train["loc_std"]   = train["geolocation_name"].map(loc_std)
train["loc_min"]   = train["geolocation_name"].map(loc_min)
train["loc_max"]   = train["geolocation_name"].map(loc_max)
train["loc_first"] = train["geolocation_name"].map(loc_first)
train["loc_last"]  = train["geolocation_name"].map(loc_last)
train["loc_slope"] = train["geolocation_name"].map(loc_slope)
train["t_first"]   = train["geolocation_name"].map(t_first)
train["t_last"]    = train["geolocation_name"].map(t_last)

# map to test
test["loc_mean"]  = test["geolocation_name"].map(loc_mean)
test["loc_std"]   = test["geolocation_name"].map(loc_std)
test["loc_min"]   = test["geolocation_name"].map(loc_min)
test["loc_max"]   = test["geolocation_name"].map(loc_max)
test["loc_first"] = test["geolocation_name"].map(loc_first)
test["loc_last"]  = test["geolocation_name"].map(loc_last)
test["loc_slope"] = test["geolocation_name"].map(loc_slope)
test["t_first"]   = test["geolocation_name"].map(t_first)
test["t_last"]    = test["geolocation_name"].map(t_last)

# =========================
# DELTA TIME FEATURE
# =========================
train["delta_t"] = train["time_idx"] - train["t_last"]
test["delta_t"]  = test["time_idx"]  - test["t_last"]

# =========================
# FILL ANY MISSING (SAFETY)
# =========================
for df in [train, test]:
    for c in ["loc_mean","loc_std","loc_min","loc_max","loc_first","loc_last","loc_slope","delta_t"]:
        df[c] = df[c].fillna(df[c].median())

# =========================
# FEATURES
# =========================
features = [
    "time_idx",
    "delta_t",
    "is_japan",
    "straight_distance_to_capital_km",
    "loc_mean","loc_std","loc_min","loc_max",
    "loc_first","loc_last","loc_slope"
]

X = train[features]
X_test = test[features]

# =========================
# LOG TARGET
# =========================
y = np.log1p(train[TARGET])

# =========================
# TRAIN
# =========================
print("Training model...")

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

oof = np.zeros(len(train))
test_preds = np.zeros(len(test))

for fold, (tr_idx, va_idx) in enumerate(kf.split(X)):
    print(f"\nFold {fold+1}")

    Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
    ytr, yva = y.iloc[tr_idx], y.iloc[va_idx]

    model = lgb.LGBMRegressor(
        n_estimators=4000,
        learning_rate=0.03,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=5,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=SEED + fold
    )

    model.fit(
        Xtr, ytr,
        eval_set=[(Xva, yva)],
        eval_metric="mae",
        callbacks=[lgb.early_stopping(200, verbose=False)]
    )

    pred_val = np.expm1(model.predict(Xva))
    oof[va_idx] = pred_val

    test_preds += np.expm1(model.predict(X_test)) / N_FOLDS

    fold_mae = mean_absolute_error(train[TARGET].iloc[va_idx], pred_val)
    print("  Fold MAE:", fold_mae)

# =========================
# FINAL CV
# =========================
cv = mean_absolute_error(train[TARGET], oof)
print("\n==============================")
print("FINAL CV MAE:", cv)
print("==============================")

# =========================
# CLIP & SAVE
# =========================
test_preds = np.clip(test_preds, 50, 6000)

sub = pd.DataFrame({
    "data_id": test["data_id"],
    TARGET: test_preds
})

sub.to_csv("submission_ultimate_location_time.csv", index=False)

print("\nSaved: submission_ultimate_location_time.csv")
print("Range:", test_preds.min(), test_preds.max())
print("Mean :", test_preds.mean())
print("DONE")
