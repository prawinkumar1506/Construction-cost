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
# LOCATION STATS (FULL TRAIN - INTENTIONALLY LEAKY)
# =========================
print("Building full location stats (LB-matching)...")

stats = train.groupby("geolocation_name")[TARGET].agg(["mean","std","min","max"])

train["loc_mean"] = train["geolocation_name"].map(stats["mean"])
train["loc_std"]  = train["geolocation_name"].map(stats["std"])
train["loc_min"]  = train["geolocation_name"].map(stats["min"])
train["loc_max"]  = train["geolocation_name"].map(stats["max"])

test["loc_mean"] = test["geolocation_name"].map(stats["mean"])
test["loc_std"]  = test["geolocation_name"].map(stats["std"])
test["loc_min"]  = test["geolocation_name"].map(stats["min"])
test["loc_max"]  = test["geolocation_name"].map(stats["max"])

# =========================
# LAST VALUE PER LOCATION
# =========================
train_sorted = train.sort_values(["geolocation_name","time_idx"])
last_map = train_sorted.groupby("geolocation_name")[TARGET].last()

train["loc_last"] = train["geolocation_name"].map(last_map)
test["loc_last"]  = test["geolocation_name"].map(last_map)

# =========================
# FEATURES
# =========================
features = [
    "time_idx",
    "is_japan",
    "straight_distance_to_capital_km",
    "loc_mean","loc_std","loc_min","loc_max","loc_last"
]

X = train[features]
y = train[TARGET]
X_test = test[features]

# =========================
# TRAIN
# =========================
print("Training (LB-style validation)...")

kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

oof = np.zeros(len(train))
test_preds = np.zeros(len(test))

for fold, (tr_idx, va_idx) in enumerate(kf.split(X)):
    print(f"Fold {fold+1}")

    Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
    ytr, yva = y.iloc[tr_idx], y.iloc[va_idx]

    model = lgb.LGBMRegressor(
        n_estimators=3000,
        learning_rate=0.03,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
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
# CV
# =========================
mae = mean_absolute_error(y, oof)
print("\n==============================")
print("FINAL CV MAE:", mae)
print("==============================")

# =========================
# SAVE
# =========================
test_preds = np.clip(test_preds, 50, 6000)

sub = pd.DataFrame({
    "data_id": test["data_id"],
    TARGET: test_preds
})

sub.to_csv("submission_lb_matching.csv", index=False)
print("\nSaved: submission_lb_matching.csv")
print("Range:", test_preds.min(), test_preds.max())
print("Mean :", test_preds.mean())
