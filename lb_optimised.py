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

print("="*60)
print("🎯 LB-OPTIMIZED MODEL")
print("Strategy: Enhanced location features + minimal complexity")
print("="*60)

# =========================
# LOAD
# =========================
print("\n[1/5] Loading data...")
train = pd.read_csv(TRAIN_CSV)
test  = pd.read_csv(TEST_CSV)

# =========================
# TIME FEATURES
# =========================
print("\n[2/5] Engineering time features...")

def add_time(df):
    df = df.copy()
    df["year"] = df["quarter_label"].str[:4].astype(int)
    df["q"] = df["quarter_label"].str[-1].astype(int)
    df["time_idx"] = df["year"] * 4 + df["q"]
    return df

train = add_time(train)
test  = add_time(test)

# =========================
# COUNTRY FEATURES
# =========================
train["is_japan"] = (train["country"] == "Japan").astype(int)
test["is_japan"]  = (test["country"] == "Japan").astype(int)

train["is_philippines"] = (train["country"] == "Philippines").astype(int)
test["is_philippines"]  = (test["country"] == "Philippines").astype(int)

# =========================
# LOCATION STATS (FULL TRAIN - LB MATCHING)
# =========================
print("\n[3/5] Building location features...")

train_sorted = train.sort_values(["geolocation_name","time_idx"])

# Full statistics per location
stats = train.groupby("geolocation_name")[TARGET].agg([
    "mean", "std", "min", "max", "median",
    ("q25", lambda x: x.quantile(0.25)),
    ("q75", lambda x: x.quantile(0.75))
])

# Last and first values
grp = train_sorted.groupby("geolocation_name")
loc_last = grp[TARGET].last()
loc_first = grp[TARGET].first()
t_last = grp["time_idx"].last()
t_first = grp["time_idx"].first()

# Slope (trend)
loc_slope = (loc_last - loc_first) / (t_last - t_first).replace(0, 1)

# Map to train
for col in ["mean", "std", "min", "max", "median", "q25", "q75"]:
    train[f"loc_{col}"] = train["geolocation_name"].map(stats[col])
    test[f"loc_{col}"] = test["geolocation_name"].map(stats[col])

train["loc_last"] = train["geolocation_name"].map(loc_last)
train["loc_first"] = train["geolocation_name"].map(loc_first)
train["loc_slope"] = train["geolocation_name"].map(loc_slope)

test["loc_last"] = test["geolocation_name"].map(loc_last)
test["loc_first"] = test["geolocation_name"].map(loc_first)
test["loc_slope"] = test["geolocation_name"].map(loc_slope)

# Delta time
train["delta_t"] = train["time_idx"] - train["geolocation_name"].map(t_last)
test["delta_t"] = test["time_idx"] - test["geolocation_name"].map(t_last)

# =========================
# ADDITIONAL SMART FEATURES
# =========================
print("\n[4/5] Creating interaction features...")

# Range (how much variation in location)
train["loc_range"] = train["loc_max"] - train["loc_min"]
test["loc_range"] = test["loc_max"] - test["loc_min"]

# IQR (interquartile range)
train["loc_iqr"] = train["loc_q75"] - train["loc_q25"]
test["loc_iqr"] = test["loc_q75"] - test["loc_q25"]

# Coefficient of variation
train["loc_cv"] = train["loc_std"] / train["loc_mean"]
test["loc_cv"] = test["loc_std"] / test["loc_mean"]

# Distance from mean
train["dist_from_mean"] = np.abs(train["loc_last"] - train["loc_mean"])
test["dist_from_mean"] = np.abs(test["loc_last"] - test["loc_mean"])

# Trend extrapolation
train["trend_pred"] = train["loc_last"] + train["loc_slope"] * train["delta_t"]
test["trend_pred"] = test["loc_last"] + test["loc_slope"] * test["delta_t"]

# Distance * location mean (logistics cost proxy)
train["dist_loc"] = train["straight_distance_to_capital_km"] * train["loc_mean"]
test["dist_loc"] = test["straight_distance_to_capital_km"] * test["loc_mean"]

# Time * slope (acceleration)
train["time_slope"] = train["time_idx"] * train["loc_slope"]
test["time_slope"] = test["time_idx"] * test["loc_slope"]

# Fill any NaNs
for df in [train, test]:
    df.fillna(df.median(numeric_only=True), inplace=True)

# =========================
# FEATURES
# =========================
features = [
    # Time
    "time_idx",
    "delta_t",
    
    # Country
    "is_japan",
    "is_philippines",
    
    # Distance
    "straight_distance_to_capital_km",
    
    # Location core stats
    "loc_mean", "loc_std", "loc_min", "loc_max", "loc_median",
    "loc_q25", "loc_q75",
    "loc_first", "loc_last", "loc_slope",
    
    # Location derived
    "loc_range", "loc_iqr", "loc_cv",
    
    # Smart features
    "dist_from_mean", "trend_pred",
    "dist_loc", "time_slope"
]

X = train[features]
y = train[TARGET]
X_test = test[features]

print(f"\nTotal features: {len(features)}")

# =========================
# TRAIN MULTIPLE MODELS
# =========================
print("\n[5/5] Training models...")
print("-" * 60)

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

# Model 1: Standard LightGBM
print("\nModel 1: Standard LightGBM")
oof_1 = np.zeros(len(train))
test_preds_1 = np.zeros(len(test))

for fold, (tr_idx, va_idx) in enumerate(kf.split(X)):
    Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
    ytr, yva = y.iloc[tr_idx], y.iloc[va_idx]
    
    model = lgb.LGBMRegressor(
        n_estimators=3000,
        learning_rate=0.03,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED + fold,
        verbose=-1
    )
    
    model.fit(Xtr, ytr, eval_set=[(Xva, yva)],
              callbacks=[lgb.early_stopping(100, verbose=False)])
    
    oof_1[va_idx] = model.predict(Xva)
    test_preds_1 += model.predict(X_test) / N_FOLDS

mae_1 = mean_absolute_error(y, oof_1)
print(f"  CV MAE: {mae_1:.2f}")

# Model 2: Deeper trees
print("\nModel 2: Deeper LightGBM")
oof_2 = np.zeros(len(train))
test_preds_2 = np.zeros(len(test))

for fold, (tr_idx, va_idx) in enumerate(kf.split(X)):
    Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
    ytr, yva = y.iloc[tr_idx], y.iloc[va_idx]
    
    model = lgb.LGBMRegressor(
        n_estimators=4000,
        learning_rate=0.02,
        num_leaves=96,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_samples=3,
        reg_alpha=0.05,
        reg_lambda=0.05,
        random_state=SEED + fold + 100,
        verbose=-1
    )
    
    model.fit(Xtr, ytr, eval_set=[(Xva, yva)],
              callbacks=[lgb.early_stopping(150, verbose=False)])
    
    oof_2[va_idx] = model.predict(Xva)
    test_preds_2 += model.predict(X_test) / N_FOLDS

mae_2 = mean_absolute_error(y, oof_2)
print(f"  CV MAE: {mae_2:.2f}")

# Model 3: Conservative regularization
print("\nModel 3: Regularized LightGBM")
oof_3 = np.zeros(len(train))
test_preds_3 = np.zeros(len(test))

for fold, (tr_idx, va_idx) in enumerate(kf.split(X)):
    Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
    ytr, yva = y.iloc[tr_idx], y.iloc[va_idx]
    
    model = lgb.LGBMRegressor(
        n_estimators=2500,
        learning_rate=0.04,
        num_leaves=48,
        subsample=0.75,
        colsample_bytree=0.75,
        min_child_samples=10,
        reg_alpha=0.2,
        reg_lambda=0.2,
        random_state=SEED + fold + 200,
        verbose=-1
    )
    
    model.fit(Xtr, ytr, eval_set=[(Xva, yva)],
              callbacks=[lgb.early_stopping(100, verbose=False)])
    
    oof_3[va_idx] = model.predict(Xva)
    test_preds_3 += model.predict(X_test) / N_FOLDS

mae_3 = mean_absolute_error(y, oof_3)
print(f"  CV MAE: {mae_3:.2f}")

# =========================
# FIND OPTIMAL ENSEMBLE
# =========================
print("\n" + "="*60)
print("Finding optimal 3-model ensemble...")
print("-" * 60)

best_mae = float('inf')
best_weights = None

for w1 in np.arange(0, 1.05, 0.1):
    for w2 in np.arange(0, 1.05 - w1, 0.1):
        w3 = 1 - w1 - w2
        if w3 < 0:
            continue
            
        oof_ens = w1 * oof_1 + w2 * oof_2 + w3 * oof_3
        mae = mean_absolute_error(y, oof_ens)
        
        if mae < best_mae:
            best_mae = mae
            best_weights = (w1, w2, w3)

print(f"\n🎯 Best weights: M1={best_weights[0]:.1f}, M2={best_weights[1]:.1f}, M3={best_weights[2]:.1f}")
print(f"🏆 Best Ensemble CV MAE: {best_mae:.2f}")

# =========================
# GENERATE SUBMISSIONS
# =========================
print("\n" + "="*60)
print("Generating submissions...")
print("-" * 60)

# Clip predictions
test_preds_1 = np.clip(test_preds_1, 50, 6000)
test_preds_2 = np.clip(test_preds_2, 50, 6000)
test_preds_3 = np.clip(test_preds_3, 50, 6000)

# Ensemble
test_preds_ens = best_weights[0] * test_preds_1 + \
                 best_weights[1] * test_preds_2 + \
                 best_weights[2] * test_preds_3
test_preds_ens = np.clip(test_preds_ens, 50, 6000)

# Save
sub_1 = pd.DataFrame({"data_id": test["data_id"], TARGET: test_preds_1})
sub_2 = pd.DataFrame({"data_id": test["data_id"], TARGET: test_preds_2})
sub_3 = pd.DataFrame({"data_id": test["data_id"], TARGET: test_preds_3})
sub_ens = pd.DataFrame({"data_id": test["data_id"], TARGET: test_preds_ens})

sub_1.to_csv("submission_lb_opt_m1.csv", index=False)
sub_2.to_csv("submission_lb_opt_m2.csv", index=False)
sub_3.to_csv("submission_lb_opt_m3.csv", index=False)
sub_ens.to_csv("submission_lb_opt_ensemble.csv", index=False)

print("\n✅ Saved 4 submissions:")
print("   - submission_lb_opt_m1.csv")
print("   - submission_lb_opt_m2.csv")
print("   - submission_lb_opt_m3.csv")
print("   - submission_lb_opt_ensemble.csv")

# =========================
# SUMMARY
# =========================
print("\n" + "="*60)
print("🏆 FINAL RESULTS")
print("="*60)
print(f"Model 1 CV MAE: {mae_1:.2f}")
print(f"Model 2 CV MAE: {mae_2:.2f}")
print(f"Model 3 CV MAE: {mae_3:.2f}")
print(f"Ensemble CV MAE: {best_mae:.2f}")
print(f"\nEnsemble: {best_weights[0]:.1f}*M1 + {best_weights[1]:.1f}*M2 + {best_weights[2]:.1f}*M3")
print(f"\nTest predictions mean: {test_preds_ens.mean():.2f}")
print("="*60)
print("\n🎯 Try submitting the ensemble first!")
print("📊 If it doesn't beat 0.231, try individual models")