"""
HYBRID BREAKTHROUGH MODEL - 0.229 BASE + JAPAN OPTIMIZATION
Combine proven 0.229 pipeline with ultra-optimized Japan model
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor
import xgboost as xgb
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings("ignore")

TARGET = "construction_cost_per_m2_usd"
TRAIN_CSV = "datasets/train_tabular.csv"
TEST_CSV = "datasets/evaluation_tabular_no_target.csv"
N_FOLDS = 3
SEEDS = [42, 777, 1234]

print("="*80)
print("HYBRID BREAKTHROUGH MODEL - 0.229 BASE + JAPAN OPTIMIZATION")
print("="*80)

# Load data
train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)

def add_time_features(df):
    df = df.copy()
    df["year"] = df["quarter_label"].str[:4].astype(int)
    df["quarter"] = df["quarter_label"].str[-1].astype(int)
    df["quarter_sin"] = np.sin(2 * np.pi * df["quarter"] / 4)
    df["quarter_cos"] = np.cos(2 * np.pi * df["quarter"] / 4)
    df["year_quarter"] = df["year"] * 4 + df["quarter"]
    df["time_index"] = (df["year"] - df["year"].min()) * 4 + df["quarter"]
    return df

train = add_time_features(train)
test = add_time_features(test)

# External data (simplified for speed)
try:
    comm_raw = pd.read_csv("external/commodity_prices.csv")
    comm_raw['date'] = pd.to_datetime(comm_raw['date'])
    comm_raw['year'] = comm_raw['date'].dt.year
    oil_yearly = comm_raw.groupby('year')['oil_brent'].mean().reset_index()
    oil_yearly.columns = ['year', 'oil_price']
    train = train.merge(oil_yearly, on="year", how="left")
    test = test.merge(oil_yearly, on="year", how="left")
    train['oil_price'] = train['oil_price'].fillna(50)
    test['oil_price'] = test['oil_price'].fillna(50)
except:
    train['oil_price'] = 50.0
    test['oil_price'] = 50.0

# ULTRA-OPTIMIZED JAPAN MODEL
print("\n[JAPAN OPTIMIZATION] Creating ultra-optimized Japan model...")

japan_train = train[train['country'] == 'Japan'].copy()
japan_test = test[test['country'] == 'Japan'].copy()

# Remove Japan outliers
iso_forest = IsolationForest(contamination=0.05, random_state=42)
outlier_mask = iso_forest.fit_predict(japan_train[[TARGET, 'deflated_gdp_usd']]) == -1
japan_train_clean = japan_train[~outlier_mask].copy()

print(f"Japan: Removed {outlier_mask.sum()} outliers, {len(japan_train_clean)} clean samples")

# Japan-specific location features
loc_stats = japan_train_clean.groupby("geolocation_name")[TARGET].agg([
    "mean", "std", "count", "min", "max"
]).fillna(0)

japan_train_sorted = japan_train_clean.sort_values(["geolocation_name", "time_index"])
grp = japan_train_sorted.groupby("geolocation_name")
loc_trend = (grp[TARGET].last() - grp[TARGET].first()) / (grp["time_index"].last() - grp["time_index"].first()).replace(0, 1)

# Map Japan features
for col in ["mean", "std", "count", "min", "max"]:
    japan_train_clean[f"japan_loc_{col}"] = japan_train_clean["geolocation_name"].map(loc_stats[col])
    japan_test[f"japan_loc_{col}"] = japan_test["geolocation_name"].map(loc_stats[col]).fillna(0)

japan_train_clean["japan_loc_trend"] = japan_train_clean["geolocation_name"].map(loc_trend)
japan_test["japan_loc_trend"] = japan_test["geolocation_name"].map(loc_trend).fillna(0)

# Japan economic features
japan_gdp_median = japan_train_clean['deflated_gdp_usd'].median()
japan_train_clean['japan_gdp_norm'] = japan_train_clean['deflated_gdp_usd'] / japan_gdp_median
japan_test['japan_gdp_norm'] = japan_test['deflated_gdp_usd'] / japan_gdp_median

# Japan features
japan_features = [
    'time_index', 'quarter_sin', 'quarter_cos',
    'japan_gdp_norm', 'us_cpi', 'oil_price',
    'straight_distance_to_capital_km',
    'japan_loc_mean', 'japan_loc_std', 'japan_loc_count', 'japan_loc_trend'
]

X_japan = japan_train_clean[japan_features].fillna(0)
y_japan = japan_train_clean[TARGET]
X_test_japan = japan_test[japan_features].fillna(0)
groups_japan = japan_train_clean['geolocation_name']

# Train ultra-optimized Japan model with GroupKFold
gkf = GroupKFold(n_splits=5)
japan_oof = np.zeros(len(japan_train_clean))
japan_test_preds = np.zeros(len(japan_test))

for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_japan, y_japan, groups_japan)):
    Xtr, Xva = X_japan.iloc[tr_idx], X_japan.iloc[va_idx]
    ytr, yva = y_japan.iloc[tr_idx], y_japan.iloc[va_idx]
    
    model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.01,  # Very conservative
        max_depth=4,
        num_leaves=15,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=0.5,  # Heavy regularization
        reg_lambda=0.5,
        random_state=42 + fold,
        verbose=-1
    )
    
    model.fit(Xtr, ytr, eval_set=[(Xva, yva)], 
             callbacks=[lgb.early_stopping(200, verbose=False)])
    
    japan_oof[va_idx] = model.predict(Xva)
    japan_test_preds += model.predict(X_test_japan) / 5

japan_cv = mean_absolute_error(y_japan, japan_oof)
print(f"Ultra-optimized Japan CV: {japan_cv:.2f}")

# PROVEN 0.229 PIPELINE FOR FULL DATASET
print("\n[GLOBAL MODEL] Running proven 0.229 pipeline...")

# Location features (global)
stats = train.groupby("geolocation_name")[TARGET].agg([
    "mean", "std", "min", "max", "median", "count"
])

for col in ["mean", "std", "min", "max", "median", "count"]:
    train[f"loc_{col}"] = train["geolocation_name"].map(stats[col])
    test[f"loc_{col}"] = test["geolocation_name"].map(stats[col])

# Convert categorical
categorical_cols = ['developed_country', 'landlocked', 'access_to_airport', 
                   'access_to_port', 'access_to_highway', 'access_to_railway']

for col in categorical_cols:
    if col in train.columns and train[col].dtype == 'object':
        train[col] = (train[col] == 'Yes').astype(int)
        test[col] = (test[col] == 'Yes').astype(int)

# Global features
global_features = [
    'year', 'quarter', 'quarter_sin', 'quarter_cos', 'year_quarter',
    'deflated_gdp_usd', 'us_cpi', 'straight_distance_to_capital_km',
    'developed_country', 'landlocked',
    'access_to_airport', 'access_to_port', 'access_to_highway', 'access_to_railway',
    'oil_price', 'loc_mean', 'loc_std', 'loc_min', 'loc_max', 'loc_median', 'loc_count'
]

features = [f for f in global_features if f in train.columns and f in test.columns]
X = train[features].fillna(0)
y = train[TARGET]
X_test = test[features].fillna(0)

print(f"Global model features: {len(features)}")

# Multi-seed ensemble (proven 0.229 approach)
all_oof_cat = []
all_test_cat = []

for seed in SEEDS:
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    oof_cat = np.zeros(len(train))
    test_preds_cat = np.zeros(len(test))
    
    X_cat = X.copy()
    X_test_cat = X_test.copy()
    X_cat['country'] = train['country']
    X_cat['geolocation_name'] = train['geolocation_name']
    X_test_cat['country'] = test['country']
    X_test_cat['geolocation_name'] = test['geolocation_name']
    
    cat_indices = [X_cat.columns.get_loc('country'), X_cat.columns.get_loc('geolocation_name')]
    
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_cat)):
        Xtr, Xva = X_cat.iloc[tr_idx], X_cat.iloc[va_idx]
        ytr, yva = y.iloc[tr_idx], y.iloc[va_idx]
        
        model = CatBoostRegressor(
            iterations=1000,
            learning_rate=0.05,
            depth=7,
            l2_leaf_reg=2,
            random_seed=seed + fold,
            verbose=0,
            early_stopping_rounds=100
        )
        
        model.fit(Xtr, ytr, cat_features=cat_indices, 
                 eval_set=(Xva, yva), verbose=False)
        
        oof_cat[va_idx] = model.predict(Xva)
        test_preds_cat += model.predict(X_test_cat) / N_FOLDS
    
    all_oof_cat.append(oof_cat)
    all_test_cat.append(test_preds_cat)

# Average global model
global_oof = np.mean(all_oof_cat, axis=0)
global_test = np.mean(all_test_cat, axis=0)
global_cv = mean_absolute_error(y, global_oof)

print(f"Global CatBoost CV: {global_cv:.2f}")

# HYBRID COMBINATION
print("\n[HYBRID COMBINATION] Combining models...")

# Create hybrid predictions
hybrid_test = global_test.copy()

# Replace Japan predictions with ultra-optimized model
japan_mask = test['country'] == 'Japan'
hybrid_test[japan_mask] = japan_test_preds

# Calculate hybrid CV
hybrid_oof = global_oof.copy()
japan_train_mask = train['country'] == 'Japan'
japan_clean_indices = japan_train[~outlier_mask].index
hybrid_oof[japan_clean_indices] = japan_oof

# For outliers, keep global predictions
hybrid_cv = mean_absolute_error(y, hybrid_oof)

print(f"Hybrid CV: {hybrid_cv:.2f}")
print(f"Japan samples replaced: {japan_mask.sum()}")

# ENSEMBLE WITH BEST EXISTING MODEL
print("\n[FINAL ENSEMBLE] Creating ultimate hybrid...")

try:
    # Load best existing model (0.2296)
    sub_winner = pd.read_csv("submission_ultimate_018.csv")
    pred_winner = sub_winner[TARGET].values
    
    # Multiple blend strategies
    blends = {
        "conservative": 0.8 * pred_winner + 0.2 * hybrid_test,
        "balanced": 0.6 * pred_winner + 0.4 * hybrid_test,
        "aggressive": 0.4 * pred_winner + 0.6 * hybrid_test,
        "pure_hybrid": hybrid_test
    }
    
    for name, pred in blends.items():
        pred_clipped = np.clip(pred, 50, 6000)
        submission = pd.DataFrame({
            "data_id": test["data_id"],
            TARGET: pred_clipped
        })
        submission.to_csv(f"submission_hybrid_018_{name}.csv", index=False)
        print(f"✓ submission_hybrid_018_{name}.csv (mean: {pred_clipped.mean():.0f})")
        
except:
    # Fallback to pure hybrid
    pred_clipped = np.clip(hybrid_test, 50, 6000)
    submission = pd.DataFrame({
        "data_id": test["data_id"],
        TARGET: pred_clipped
    })
    submission.to_csv("submission_hybrid_018_pure.csv", index=False)
    print(f"✓ submission_hybrid_018_pure.csv (mean: {pred_clipped.mean():.0f})")

# Performance prediction
if hybrid_cv < 85:
    expected_lb = "0.18-0.21"
    status = "🎯 BREAKTHROUGH ACHIEVED!"
elif hybrid_cv < 90:
    expected_lb = "0.20-0.23"
    status = "🔥 EXCELLENT PROGRESS!"
else:
    expected_lb = "0.22-0.25"
    status = "📈 GOOD IMPROVEMENT"

print(f"\n" + "="*80)
print(f"HYBRID BREAKTHROUGH MODEL COMPLETE")
print(f"="*80)
print(f"Global CV: {global_cv:.2f}")
print(f"Japan CV: {japan_cv:.2f}")
print(f"Hybrid CV: {hybrid_cv:.2f}")
print(f"Expected LB: {expected_lb}")
print(f"Status: {status}")
print(f"\nSubmission Priority:")
print(f"1. submission_hybrid_018_conservative.csv (SAFEST)")
print(f"2. submission_hybrid_018_balanced.csv (BALANCED)")
print(f"3. submission_hybrid_018_pure.csv (PURE HYBRID)")
print(f"\nKey Innovations:")
print(f"✓ Proven 0.229 pipeline as base")
print(f"✓ Ultra-optimized Japan model (CV: {japan_cv:.1f})")
print(f"✓ Outlier removal for Japan")
print(f"✓ GroupKFold validation for Japan")
print(f"✓ Heavy regularization for Japan")
print(f"✓ Hybrid combination strategy")
print("="*80)