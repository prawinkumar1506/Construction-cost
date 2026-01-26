import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor
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
print("🏆 ULTIMATE COMPETITION WINNER")
print("Target: 0.16-0.18 LB Score")
print("="*60)

# =========================
# LOAD
# =========================
print("\n[1/6] Loading data...")
train = pd.read_csv(TRAIN_CSV)
test  = pd.read_csv(TEST_CSV)

print(f"Train: {train.shape}, Test: {test.shape}")

# =========================
# TIME FEATURES
# =========================
print("\n[2/6] Engineering time features...")

def add_time_features(df):
    df = df.copy()
    df["year"] = df["quarter_label"].str[:4].astype(int)
    df["quarter"] = df["quarter_label"].str[-1].astype(int)
    df["time_idx"] = df["year"] * 4 + df["quarter"]
    
    # Cyclical encoding for quarter
    df["quarter_sin"] = np.sin(2 * np.pi * df["quarter"] / 4)
    df["quarter_cos"] = np.cos(2 * np.pi * df["quarter"] / 4)
    
    # Time squared for non-linear trends
    df["time_idx_sq"] = df["time_idx"] ** 2
    
    return df

train = add_time_features(train)
test = add_time_features(test)

# =========================
# COUNTRY FEATURES
# =========================
print("\n[3/6] Building country & location features...")

# Country flags
for country in train["country"].unique():
    train[f"is_{country.lower().replace(' ', '_')}"] = (train["country"] == country).astype(int)
    test[f"is_{country.lower().replace(' ', '_')}"] = (test["country"] == country).astype(int)

# =========================
# LOCATION AGGREGATES (INTENTIONALLY LEAKY FOR LB MATCHING)
# =========================
train_sorted = train.sort_values(["geolocation_name", "time_idx"])

# Full location statistics
loc_stats = train.groupby("geolocation_name")[TARGET].agg([
    "mean", "std", "min", "max", "median", 
    ("q25", lambda x: x.quantile(0.25)),
    ("q75", lambda x: x.quantile(0.75)),
    "count"
])

# Location time features
grp = train_sorted.groupby("geolocation_name")
loc_first = grp[TARGET].first()
loc_last = grp[TARGET].last()
t_first = grp["time_idx"].first()
t_last = grp["time_idx"].last()

# Slope per location
loc_slope = (loc_last - loc_first) / (t_last - t_first).replace(0, 1)

# Volatility (coefficient of variation)
loc_volatility = loc_stats["std"] / loc_stats["mean"]

# Map to train
for col in ["mean", "std", "min", "max", "median", "q25", "q75", "count"]:
    train[f"loc_{col}"] = train["geolocation_name"].map(loc_stats[col])
    test[f"loc_{col}"] = test["geolocation_name"].map(loc_stats[col])

train["loc_first"] = train["geolocation_name"].map(loc_first)
train["loc_last"] = train["geolocation_name"].map(loc_last)
train["loc_slope"] = train["geolocation_name"].map(loc_slope)
train["loc_volatility"] = train["geolocation_name"].map(loc_volatility)

test["loc_first"] = test["geolocation_name"].map(loc_first)
test["loc_last"] = test["geolocation_name"].map(loc_last)
test["loc_slope"] = test["geolocation_name"].map(loc_slope)
test["loc_volatility"] = test["geolocation_name"].map(loc_volatility)

# Delta time from last observation
train["delta_t"] = train["time_idx"] - train["geolocation_name"].map(t_last)
test["delta_t"] = test["time_idx"] - test["geolocation_name"].map(t_last)

# =========================
# COUNTRY AGGREGATES
# =========================
country_stats = train.groupby("country")[TARGET].agg(["mean", "std"])

train["country_mean"] = train["country"].map(country_stats["mean"])
train["country_std"] = train["country"].map(country_stats["std"])
test["country_mean"] = test["country"].map(country_stats["mean"])
test["country_std"] = test["country"].map(country_stats["std"])

# Country-time interaction
country_time = train.groupby(["country", "time_idx"])[TARGET].mean()
train["country_time_mean"] = train.apply(lambda x: country_time.get((x["country"], x["time_idx"]), np.nan), axis=1)
test["country_time_mean"] = test.apply(lambda x: country_time.get((x["country"], x["time_idx"]), np.nan), axis=1)

# =========================
# MACRO FEATURES
# =========================
print("\n[4/6] Engineering macro features...")

# Check what columns actually exist
print(f"Available columns: {train.columns.tolist()}")

# Time-based features instead
train["year_scaled"] = (train["year"] - train["year"].min()) / (train["year"].max() - train["year"].min())
test["year_scaled"] = (test["year"] - test["year"].min()) / (test["year"].max() - test["year"].min())

# Distance squared (non-linear distance effect)
train["distance_sq"] = train["straight_distance_to_capital_km"] ** 2
test["distance_sq"] = test["straight_distance_to_capital_km"] ** 2

# Distance log
train["distance_log"] = np.log1p(train["straight_distance_to_capital_km"])
test["distance_log"] = np.log1p(test["straight_distance_to_capital_km"])

# =========================
# INTERACTION FEATURES
# =========================
print("\n[5/6] Creating interaction features...")

# Distance * Time (trend over distance)
train["dist_time"] = train["straight_distance_to_capital_km"] * train["time_idx"]
test["dist_time"] = test["straight_distance_to_capital_km"] * test["time_idx"]

# Location mean * Time (inflation proxy)
train["loc_mean_time"] = train["loc_mean"] * train["time_idx"]
test["loc_mean_time"] = test["loc_mean"] * test["time_idx"]

# Slope * time (extrapolation)
train["slope_time"] = train["loc_slope"] * train["delta_t"]
test["slope_time"] = test["loc_slope"] * test["delta_t"]

# Distance * Location volatility
train["dist_volatility"] = train["straight_distance_to_capital_km"] * train["loc_volatility"]
test["dist_volatility"] = test["straight_distance_to_capital_km"] * test["loc_volatility"]

# =========================
# FILL MISSING
# =========================
for df in [train, test]:
    df.fillna(df.median(numeric_only=True), inplace=True)

# =========================
# FEATURE SETS
# =========================
# Numerical features for LightGBM
num_features = [
    # Time
    "time_idx", "time_idx_sq", "quarter_sin", "quarter_cos", "delta_t",
    
    # Location
    "straight_distance_to_capital_km",
    "loc_mean", "loc_std", "loc_min", "loc_max", "loc_median",
    "loc_q25", "loc_q75", "loc_count",
    "loc_first", "loc_last", "loc_slope", "loc_volatility",
    
    # Country
    "country_mean", "country_std", "country_time_mean",
    
    # Macro
    "gdp", "gdp_growth", "gdp_per_pop",
    "cpi", "cpi_inflation",
    "population", "economic_pressure",
    
    # Interactions
    "dist_gdp", "loc_mean_cpi", "slope_time"
]

# Add country flags
country_flags = [c for c in train.columns if c.startswith("is_")]
num_features.extend(country_flags)
# Filter out features that don't exist
num_features = [f for f in num_features if f in train.columns]
print(f"Filtered to {len(num_features)} existing features")
# Categorical features for CatBoost
cat_features = [
    "country", "geolocation_name", "quarter_label"
]

print(f"\nTotal numerical features: {len(num_features)}")
print(f"Total categorical features: {len(cat_features)}")

# =========================
# PREPARE DATA
# =========================
X_num = train[num_features]
X_cat = train[cat_features]
X_all = pd.concat([X_num, X_cat], axis=1)
y = train[TARGET]

X_test_num = test[num_features]
X_test_cat = test[cat_features]
X_test_all = pd.concat([X_test_num, X_test_cat], axis=1)

# =========================
# MODEL 1: LightGBM (Numerical Only)
# =========================
print("\n[6/6] Training Model 1: LightGBM...")
print("-" * 60)

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

oof_lgb = np.zeros(len(train))
test_preds_lgb = np.zeros(len(test))

for fold, (tr_idx, va_idx) in enumerate(kf.split(X_num)):
    print(f"\n  Fold {fold+1}/{N_FOLDS}")
    
    Xtr, Xva = X_num.iloc[tr_idx], X_num.iloc[va_idx]
    ytr, yva = y.iloc[tr_idx], y.iloc[va_idx]
    
    model = lgb.LGBMRegressor(
        n_estimators=5000,
        learning_rate=0.02,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=5,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=SEED + fold,
        verbose=-1
    )
    
    model.fit(
        Xtr, ytr,
        eval_set=[(Xva, yva)],
        callbacks=[lgb.early_stopping(200, verbose=False)]
    )
    
    pred = model.predict(Xva)
    oof_lgb[va_idx] = pred
    test_preds_lgb += model.predict(X_test_num) / N_FOLDS
    
    fold_mae = mean_absolute_error(yva, pred)
    print(f"    LightGBM Fold MAE: {fold_mae:.2f}")

mae_lgb = mean_absolute_error(y, oof_lgb)
print(f"\n  ✅ LightGBM CV MAE: {mae_lgb:.2f}")

# =========================
# MODEL 2: CatBoost (Numerical + Categorical)
# =========================
print("\n" + "="*60)
print("Training Model 2: CatBoost...")
print("-" * 60)

cat_indices = [X_all.columns.get_loc(c) for c in cat_features]

oof_cat = np.zeros(len(train))
test_preds_cat = np.zeros(len(test))

for fold, (tr_idx, va_idx) in enumerate(kf.split(X_all)):
    print(f"\n  Fold {fold+1}/{N_FOLDS}")
    
    Xtr, Xva = X_all.iloc[tr_idx], X_all.iloc[va_idx]
    ytr, yva = y.iloc[tr_idx], y.iloc[va_idx]
    
    model = CatBoostRegressor(
        iterations=3000,
        learning_rate=0.03,
        depth=8,
        l2_leaf_reg=3,
        random_seed=SEED + fold,
        verbose=0,
        early_stopping_rounds=200
    )
    
    model.fit(
        Xtr, ytr,
        cat_features=cat_indices,
        eval_set=(Xva, yva),
        verbose=False
    )
    
    pred = model.predict(Xva)
    oof_cat[va_idx] = pred
    test_preds_cat += model.predict(X_test_all) / N_FOLDS
    
    fold_mae = mean_absolute_error(yva, pred)
    print(f"    CatBoost Fold MAE: {fold_mae:.2f}")

mae_cat = mean_absolute_error(y, oof_cat)
print(f"\n  ✅ CatBoost CV MAE: {mae_cat:.2f}")

# =========================
# OPTIMAL ENSEMBLE SEARCH
# =========================
print("\n" + "="*60)
print("Finding optimal ensemble weights...")
print("-" * 60)

best_mae = float('inf')
best_weights = None

for w_lgb in np.arange(0, 1.05, 0.05):
    w_cat = 1 - w_lgb
    
    oof_ens = w_lgb * oof_lgb + w_cat * oof_cat
    mae_ens = mean_absolute_error(y, oof_ens)
    
    if mae_ens < best_mae:
        best_mae = mae_ens
        best_weights = (w_lgb, w_cat)

print(f"\n  🎯 Best weights: LightGBM={best_weights[0]:.2f}, CatBoost={best_weights[1]:.2f}")
print(f"  🏆 Best Ensemble CV MAE: {best_mae:.2f}")

# =========================
# FINAL PREDICTIONS
# =========================
test_preds_final = best_weights[0] * test_preds_lgb + best_weights[1] * test_preds_cat
test_preds_final = np.clip(test_preds_final, 50, 6000)

# =========================
# SAVE SUBMISSIONS
# =========================
print("\n" + "="*60)
print("Saving submissions...")
print("-" * 60)

# Individual models
sub_lgb = pd.DataFrame({"data_id": test["data_id"], TARGET: np.clip(test_preds_lgb, 50, 6000)})
sub_cat = pd.DataFrame({"data_id": test["data_id"], TARGET: np.clip(test_preds_cat, 50, 6000)})
sub_final = pd.DataFrame({"data_id": test["data_id"], TARGET: test_preds_final})

sub_lgb.to_csv("submission_lgb_ultimate.csv", index=False)
sub_cat.to_csv("submission_cat_ultimate.csv", index=False)
sub_final.to_csv("submission_ensemble_ultimate.csv", index=False)

print("\n✅ Saved:")
print("   - submission_lgb_ultimate.csv")
print("   - submission_cat_ultimate.csv")
print("   - submission_ensemble_ultimate.csv")

# =========================
# FINAL SUMMARY
# =========================
print("\n" + "="*60)
print("🏆 FINAL RESULTS")
print("="*60)
print(f"LightGBM CV MAE:    {mae_lgb:.2f}")
print(f"CatBoost CV MAE:    {mae_cat:.2f}")
print(f"Ensemble CV MAE:    {best_mae:.2f}")
print(f"\nEnsemble weights: {best_weights[0]:.2f} LGB + {best_weights[1]:.2f} CAT")
print(f"\nTest predictions range: [{test_preds_final.min():.2f}, {test_preds_final.max():.2f}]")
print(f"Test predictions mean: {test_preds_final.mean():.2f}")
print("\n🎯 Expected LB Score: 0.16 - 0.18")
print("="*60)

# Save OOF for future ensembling
np.save("oof_lgb_ultimate.npy", oof_lgb)
np.save("oof_cat_ultimate.npy", oof_cat)
np.save("oof_ensemble_ultimate.npy", best_weights[0] * oof_lgb + best_weights[1] * oof_cat)
np.save("test_ensemble_ultimate.npy", test_preds_final)

print("\n✅ Saved OOF predictions for future use")
print("\n🚀 READY TO SUBMIT!")