import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
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
    df["q"] = df["quarter_label"].str[-2:].map({"Q1":1,"Q2":2,"Q3":3,"Q4":4})
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
# PREF CODE (SAFE)
# =========================
train["pref_code"] = train["geolocation_name"].str.extract(r"(\d+)")
test["pref_code"]  = test["geolocation_name"].str.extract(r"(\d+)")

train["pref_code"] = train["pref_code"].fillna(-1).astype(int)
test["pref_code"]  = test["pref_code"].fillna(-1).astype(int)

# =========================
# CV-SAFE LOCATION ENCODING
# =========================
print("Building CV-safe location features...")

gkf = GroupKFold(n_splits=5)
groups = train["geolocation_name"]

for c in ["loc_mean","loc_std","loc_min","loc_max"]:
    train[c] = np.nan

for fold, (tr_idx, va_idx) in enumerate(gkf.split(train, train[TARGET], groups)):
    tr = train.iloc[tr_idx]
    va = train.iloc[va_idx]

    stats = tr.groupby("geolocation_name")[TARGET].agg(["mean","std","min","max"])

    train.loc[va_idx, "loc_mean"] = va["geolocation_name"].map(stats["mean"])
    train.loc[va_idx, "loc_std"]  = va["geolocation_name"].map(stats["std"])
    train.loc[va_idx, "loc_min"]  = va["geolocation_name"].map(stats["min"])
    train.loc[va_idx, "loc_max"]  = va["geolocation_name"].map(stats["max"])

# Fill missing
for c in ["loc_mean","loc_std","loc_min","loc_max"]:
    train[c] = train[c].fillna(train[TARGET].mean())

# =========================
# FULL STATS FOR TEST
# =========================
full_stats = train.groupby("geolocation_name")[TARGET].agg(["mean","std","min","max"])

test["loc_mean"] = test["geolocation_name"].map(full_stats["mean"])
test["loc_std"]  = test["geolocation_name"].map(full_stats["std"])
test["loc_min"]  = test["geolocation_name"].map(full_stats["min"])
test["loc_max"]  = test["geolocation_name"].map(full_stats["max"])

for c in ["loc_mean","loc_std","loc_min","loc_max"]:
    test[c] = test[c].fillna(train[TARGET].mean())

# =========================
# FEATURES
# =========================
drop_cols = [TARGET, "data_id", "geolocation_name", "quarter_label", "country"]
features = [c for c in train.columns if c not in drop_cols]

X = train[features]
X_test = test[features]

# keep only numeric
X = X.select_dtypes(include=[np.number])
X_test = X_test[X.columns]

y = train[TARGET]

# =========================
# TRAIN
# =========================
print("Training single global model...")

oof = np.zeros(len(train))
test_preds = np.zeros(len(test))

for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups)):
    print(f"Fold {fold+1}")

    Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
    ytr, yva = y.iloc[tr_idx], y.iloc[va_idx]

    model = lgb.LGBMRegressor(
        n_estimators=3000,
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
        callbacks=[lgb.early_stopping(100, verbose=False)]
    )

    pred = model.predict(Xva)
    oof[va_idx] = pred
    test_preds += model.predict(X_test) / N_FOLDS

    print("  Fold MAE:", mean_absolute_error(yva, pred))

# =========================
# FINAL
# =========================
mae = mean_absolute_error(y, oof)
print("\n==============================")
print("FINAL CV MAE:", mae)
print("==============================")

test_preds = np.clip(test_preds, 50, 6000)

sub = pd.DataFrame({
    "data_id": test["data_id"],
    TARGET: test_preds
})

sub.to_csv("submission_magic_location_global.csv", index=False)
print("\nSaved: submission_magic_location_global.csv")
print("Range:", test_preds.min(), test_preds.max())
print("Mean :", test_preds.mean())
