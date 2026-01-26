"""
CONSERVATIVE ENHANCEMENT - REAL LB IMPROVEMENT
Strategy: Minimal changes to proven 0.2317 approach + tiny external boost
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

TARGET = "construction_cost_per_m2_usd"
TRAIN_CSV = "datasets/train_tabular.csv"
TEST_CSV = "datasets/evaluation_tabular_no_target.csv"
N_FOLDS = 5
SEED = 42

print("="*80)
print("CONSERVATIVE ENHANCEMENT - REAL LB IMPROVEMENT")
print("Target: Beat 0.2317 with minimal overfitting")
print("="*80)

# =========================
# LOAD DATA
# =========================
print("\n[LOADING] Data...")
train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)

def add_time_features(df):
    df = df.copy()
    df["year"] = df["quarter_label"].str[:4].astype(int)
    df["quarter"] = df["quarter_label"].str[-1].astype(int)
    df["quarter_sin"] = np.sin(2 * np.pi * df["quarter"] / 4)
    df["quarter_cos"] = np.cos(2 * np.pi * df["quarter"] / 4)
    return df

train = add_time_features(train)
test = add_time_features(test)

# =========================
# MINIMAL EXTERNAL DATA
# =========================
print("\n[EXTERNAL] Adding minimal oil signal...")

try:
    comm_raw = pd.read_csv("external/commodity_prices.csv")
    comm_raw['date'] = pd.to_datetime(comm_raw['date'])
    comm_raw['year'] = comm_raw['date'].dt.year
    
    oil_yearly = comm_raw.groupby('year')['oil_brent'].mean().reset_index()
    oil_yearly['year'] = oil_yearly['year'].astype(int)
    
    train = train.merge(oil_yearly, on="year", how="left")
    test = test.merge(oil_yearly, on="year", how="left")
    
    train['oil_brent'] = train['oil_brent'].fillna(train['oil_brent'].mean())
    test['oil_brent'] = test['oil_brent'].fillna(train['oil_brent'].mean())
    
    print(f"   Added oil prices")
    
except:
    train['oil_brent'] = 50.0
    test['oil_brent'] = 50.0

# =========================
# PROVEN LOCATION FEATURES (CONSERVATIVE)
# =========================
print("\n[LOCATION] Conservative location features...")

train_sorted = train.sort_values(["geolocation_name", "year", "quarter"])

# ONLY the most proven location stats
stats = train.groupby("geolocation_name")[TARGET].agg([
    "mean", "std", "min", "max", "median", "count"
])

grp = train_sorted.groupby("geolocation_name")
loc_last = grp[TARGET].last()
loc_first = grp[TARGET].first()

# Simple trend
time_first = grp["year"].first() * 4 + grp["quarter"].first()
time_last = grp["year"].last() * 4 + grp["quarter"].last()
loc_slope = (loc_last - loc_first) / (time_last - time_first).replace(0, 1)

# Map ONLY proven features
for col in ["mean", "std", "min", "max", "median", "count"]:
    train[f"loc_{col}"] = train["geolocation_name"].map(stats[col])
    test[f"loc_{col}"] = test["geolocation_name"].map(stats[col])

train["loc_last"] = train["geolocation_name"].map(loc_last)
train["loc_slope"] = train["geolocation_name"].map(loc_slope)
test["loc_last"] = test["geolocation_name"].map(loc_last)
test["loc_slope"] = test["geolocation_name"].map(loc_slope)

# Convert categorical
categorical_cols = ['developed_country', 'landlocked', 'access_to_airport', 'access_to_port', 
                   'access_to_highway', 'access_to_railway']

for col in categorical_cols:
    if col in train.columns and train[col].dtype == 'object':
        train[col] = (train[col] == 'Yes').astype(int)
        test[col] = (test[col] == 'Yes').astype(int)

print(f"   Added conservative location features")

# =========================
# CONSERVATIVE FEATURE SET
# =========================
print("\n[FEATURES] Conservative feature set...")

# MINIMAL feature set (like your proven model)
features = [
    'year', 'quarter', 'quarter_sin', 'quarter_cos',
    'deflated_gdp_usd', 'us_cpi', 'straight_distance_to_capital_km',
    'developed_country', 'landlocked',
    'access_to_airport', 'access_to_port', 'access_to_highway', 'access_to_railway',
    # ONLY proven location features
    'loc_mean', 'loc_std', 'loc_min', 'loc_max', 'loc_median', 'loc_count',
    'loc_last', 'loc_slope',
    # MINIMAL external
    'oil_brent'
]

# ONE simple interaction
train['oil_x_distance'] = train['oil_brent'] * train['straight_distance_to_capital_km'] / 1000
test['oil_x_distance'] = test['oil_brent'] * test['straight_distance_to_capital_km'] / 1000
features.append('oil_x_distance')

features = [f for f in features if f in train.columns and f in test.columns]

X = train[features]
y = train[TARGET]
X_test = test[features]

print(f"   Using {len(features)} conservative features")

# =========================
# CONSERVATIVE TRAINING
# =========================
print("\n[TRAINING] Conservative ensemble...")

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

oof_lgb = np.zeros(len(train))
oof_cat = np.zeros(len(train))
test_preds_lgb = np.zeros(len(test))
test_preds_cat = np.zeros(len(test))

# CONSERVATIVE LightGBM (similar to your proven model)
print("\n   LightGBM (conservative)...")
for fold, (tr_idx, va_idx) in enumerate(kf.split(X)):
    print(f"      Fold {fold+1}/{N_FOLDS}", end=" ")
    
    Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
    ytr, yva = y.iloc[tr_idx], y.iloc[va_idx]
    
    model = lgb.LGBMRegressor(
        n_estimators=2500,
        learning_rate=0.03,
        max_depth=8,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        min_child_samples=20,
        random_state=SEED + fold,
        verbose=-1
    )
    
    model.fit(
        Xtr, ytr,
        eval_set=[(Xva, yva)],
        callbacks=[lgb.early_stopping(250, verbose=False)]
    )
    
    pred = model.predict(Xva)
    oof_lgb[va_idx] = pred
    test_preds_lgb += model.predict(X_test) / N_FOLDS
    
    fold_mae = mean_absolute_error(yva, pred)
    print(f"MAE: {fold_mae:.4f}")

# CONSERVATIVE CatBoost
print("\n   CatBoost (conservative)...")

X_cat = X.copy()
X_test_cat = X_test.copy()
X_cat['country'] = train['country']
X_cat['geolocation_name'] = train['geolocation_name']
X_test_cat['country'] = test['country']
X_test_cat['geolocation_name'] = test['geolocation_name']

cat_indices = [X_cat.columns.get_loc('country'), X_cat.columns.get_loc('geolocation_name')]

for fold, (tr_idx, va_idx) in enumerate(kf.split(X_cat)):
    print(f"      Fold {fold+1}/{N_FOLDS}", end=" ")
    
    Xtr, Xva = X_cat.iloc[tr_idx], X_cat.iloc[va_idx]
    ytr, yva = y.iloc[tr_idx], y.iloc[va_idx]
    
    model = CatBoostRegressor(
        iterations=2000,
        learning_rate=0.03,
        depth=8,
        l2_leaf_reg=3,
        random_seed=SEED + fold,
        verbose=0,
        early_stopping_rounds=250
    )
    
    model.fit(
        Xtr, ytr,
        cat_features=cat_indices,
        eval_set=(Xva, yva),
        verbose=False
    )
    
    pred = model.predict(Xva)
    oof_cat[va_idx] = pred
    test_preds_cat += model.predict(X_test_cat) / N_FOLDS
    
    fold_mae = mean_absolute_error(yva, pred)
    print(f"MAE: {fold_mae:.4f}")

cv_lgb = mean_absolute_error(y, oof_lgb)
cv_cat = mean_absolute_error(y, oof_cat)

print(f"\n   LightGBM CV MAE: {cv_lgb:.4f}")
print(f"   CatBoost CV MAE: {cv_cat:.4f}")

# Optimal ensemble
best_mae = float('inf')
best_weights = None

for w_lgb in np.arange(0, 1.05, 0.05):
    w_cat = 1 - w_lgb
    oof_ens = w_lgb * oof_lgb + w_cat * oof_cat
    mae = mean_absolute_error(y, oof_ens)
    
    if mae < best_mae:
        best_mae = mae
        best_weights = (w_lgb, w_cat)

print(f"\n   Optimal weights: {best_weights[0]:.2f} LGB + {best_weights[1]:.2f} CAT")
print(f"   ENSEMBLE CV MAE: {best_mae:.4f}")

# =========================
# RESULTS
# =========================
# Generate submission
test_preds_lgb = np.clip(test_preds_lgb, 50, 6000)
test_preds_cat = np.clip(test_preds_cat, 50, 6000)
test_preds_ens = best_weights[0] * test_preds_lgb + best_weights[1] * test_preds_cat
test_preds_ens = np.clip(test_preds_ens, 50, 6000)

test_df = pd.read_csv(TEST_CSV)
submission = pd.DataFrame({
    "data_id": test_df["data_id"], 
    TARGET: test_preds_ens
})

submission.to_csv("submission_conservative_enhancement.csv", index=False)

print(f"\nSaved: submission_conservative_enhancement.csv")
print(f"   CV MAE: {best_mae:.4f}")

if best_mae < 90:
    print(f"   Expected LB: 0.22-0.24 (realistic)")
elif best_mae < 95:
    print(f"   Expected LB: 0.23-0.25 (conservative)")
else:
    print(f"   Expected LB: 0.24-0.26 (safe)")

print(f"\nStrategy: Conservative enhancement of proven approach")
print(f"Goal: Beat 0.2317 without overfitting")

print("\n" + "="*80)
print("CONSERVATIVE ENHANCEMENT COMPLETE")
print("="*80)