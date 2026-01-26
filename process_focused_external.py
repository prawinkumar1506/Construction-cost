"""
FOCUSED EXTERNAL DATA MODEL - TARGET: BEAT 0.2317
Strategy: Use ONLY the most impactful external features + proven location features
"""

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
TEST_CSV = "datasets/evaluation_tabular_no_target.csv"
N_FOLDS = 5
SEED = 42

print("="*80)
print("🎯 FOCUSED EXTERNAL MODEL - TARGET: BEAT 0.2317")
print("="*80)

# =========================
# LOAD BASE DATA
# =========================
print("\n[STEP 1/5] Loading base data...")
train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)

# Add time features
def add_time_features(df):
    df = df.copy()
    df["year"] = df["quarter_label"].str[:4].astype(int)
    df["quarter"] = df["quarter_label"].str[-1].astype(int)
    df["quarter_sin"] = np.sin(2 * np.pi * df["quarter"] / 4)
    df["quarter_cos"] = np.cos(2 * np.pi * df["quarter"] / 4)
    return df

train = add_time_features(train)
test = add_time_features(test)

print(f"   ✅ Train: {train.shape}")
print(f"   ✅ Test: {test.shape}")

# =========================
# LOAD ONLY HIGH-IMPACT EXTERNAL DATA
# =========================
print("\n[STEP 2/5] Loading high-impact external data...")

# 1. OIL PRICES (Most correlated with construction costs)
try:
    comm_raw = pd.read_csv("external/commodity_prices.csv")
    comm_raw['date'] = pd.to_datetime(comm_raw['date'])
    comm_raw['year'] = comm_raw['date'].dt.year
    
    oil_yearly = comm_raw.groupby('year').agg({
        'oil_brent': 'mean',
        'oil_dubai': 'mean'
    }).reset_index()
    oil_yearly['year'] = oil_yearly['year'].astype(int)
    oil_yearly['oil_avg'] = (oil_yearly['oil_brent'] + oil_yearly['oil_dubai']) / 2
    
    print(f"   ✅ Oil prices: {oil_yearly.shape}")
    
except Exception as e:
    print(f"   ❌ Oil data failed: {e}")
    raise e

# 2. GDP DEFLATOR (Economic indicator)
try:
    gdp_raw = pd.read_csv("external/API_NY.GDP.DEFL.ZS_DS2_en_csv_v2_4610.csv", skiprows=4)
    year_cols = [c for c in gdp_raw.columns if c.isdigit() and int(c) >= 2010]
    
    gdp_long = gdp_raw.melt(
        id_vars=['Country Name'],
        value_vars=year_cols,
        var_name='year',
        value_name='gdp_deflator'
    )
    gdp_long.columns = ['country', 'year', 'gdp_deflator']
    gdp_long['year'] = gdp_long['year'].astype(int)
    gdp_long = gdp_long.dropna(subset=['gdp_deflator'])
    
    print(f"   ✅ GDP deflator: {gdp_long.shape}")
    
except Exception as e:
    print(f"   ❌ GDP data failed: {e}")
    raise e

# =========================
# MERGE EXTERNAL DATA
# =========================
print("\n[STEP 3/5] Merging external data...")

train['year'] = train['year'].astype(int)
test['year'] = test['year'].astype(int)

# Merge oil prices
train = train.merge(oil_yearly, on="year", how="left")
test = test.merge(oil_yearly, on="year", how="left")

# Merge GDP deflator
train = train.merge(gdp_long, on=["country", "year"], how="left")
test = test.merge(gdp_long, on=["country", "year"], how="left")

# Fill missing values
external_cols = ['oil_brent', 'oil_dubai', 'oil_avg', 'gdp_deflator']
for col in external_cols:
    if col in train.columns:
        train[col] = train[col].fillna(train[col].mean())
        test[col] = test[col].fillna(train[col].mean())

print(f"   ✅ Final shapes - Train: {train.shape}, Test: {test.shape}")

# =========================
# FOCUSED FEATURE ENGINEERING
# =========================
print("\n[STEP 4/5] Focused feature engineering...")

# 1. LOCATION FEATURES (PROVEN TO WORK)
train_sorted = train.sort_values(["geolocation_name", "year", "quarter"])

stats = train.groupby("geolocation_name")[TARGET].agg([
    "mean", "std", "min", "max", "median", "count"
])

grp = train_sorted.groupby("geolocation_name")
loc_last = grp[TARGET].last()
loc_first = grp[TARGET].first()

# Time-based slope
time_first = grp["year"].first() * 4 + grp["quarter"].first()
time_last = grp["year"].last() * 4 + grp["quarter"].last()
loc_slope = (loc_last - loc_first) / (time_last - time_first).replace(0, 1)

# Map location features
for col in ["mean", "std", "min", "max", "median", "count"]:
    train[f"loc_{col}"] = train["geolocation_name"].map(stats[col])
    test[f"loc_{col}"] = test["geolocation_name"].map(stats[col])

train["loc_last"] = train["geolocation_name"].map(loc_last)
train["loc_slope"] = train["geolocation_name"].map(loc_slope)
test["loc_last"] = test["geolocation_name"].map(loc_last)
test["loc_slope"] = test["geolocation_name"].map(loc_slope)

# 2. ECONOMIC GROWTH FEATURES (ONLY MOST IMPORTANT)
train_sorted = train.sort_values(["country", "year", "quarter"])
test_sorted = test.sort_values(["country", "year", "quarter"])

# Oil price growth (1-year)
train["oil_growth_4q"] = train_sorted.groupby("country")["oil_avg"].pct_change(4).values
test["oil_growth_4q"] = test_sorted.groupby("country")["oil_avg"].pct_change(4).values

# GDP deflator growth (1-year)
train["gdp_growth_4q"] = train_sorted.groupby("country")["gdp_deflator"].pct_change(4).values
test["gdp_growth_4q"] = test_sorted.groupby("country")["gdp_deflator"].pct_change(4).values

# Fill NaN
train["oil_growth_4q"] = train["oil_growth_4q"].fillna(0)
train["gdp_growth_4q"] = train["gdp_growth_4q"].fillna(0)
test["oil_growth_4q"] = test["oil_growth_4q"].fillna(0)
test["gdp_growth_4q"] = test["gdp_growth_4q"].fillna(0)

# 3. ONLY HIGH-IMPACT INTERACTIONS
train['oil_x_loc'] = train['oil_avg'] * train['loc_mean']
train['gdp_x_loc'] = train['gdp_deflator'] * train['loc_mean']
train['oil_x_distance'] = train['oil_avg'] * train['straight_distance_to_capital_km']

test['oil_x_loc'] = test['oil_avg'] * test['loc_mean']
test['gdp_x_loc'] = test['gdp_deflator'] * test['loc_mean']
test['oil_x_distance'] = test['oil_avg'] * test['straight_distance_to_capital_km']

print(f"   ✅ Added focused features")

# =========================
# PREPARE FEATURES
# =========================
print("\n[STEP 5/5] Preparing focused feature set...")

# Convert categorical to numeric
categorical_cols = ['developed_country', 'landlocked', 'access_to_airport', 'access_to_port', 
                   'access_to_highway', 'access_to_railway']

for col in categorical_cols:
    if col in train.columns and train[col].dtype == 'object':
        train[col] = (train[col] == 'Yes').astype(int)
        test[col] = (test[col] == 'Yes').astype(int)

# FOCUSED FEATURE SET (ONLY PROVEN + HIGH-IMPACT)
features = [
    # Base features
    'year', 'quarter', 'quarter_sin', 'quarter_cos',
    'deflated_gdp_usd', 'us_cpi', 'straight_distance_to_capital_km',
    'developed_country', 'landlocked',
    'access_to_airport', 'access_to_port', 'access_to_highway', 'access_to_railway',
    
    # Location features (PROVEN)
    'loc_mean', 'loc_std', 'loc_min', 'loc_max', 'loc_median', 'loc_count',
    'loc_last', 'loc_slope',
    
    # External features (HIGH-IMPACT ONLY)
    'oil_avg', 'gdp_deflator', 'oil_growth_4q', 'gdp_growth_4q',
    
    # Interactions (SELECTIVE)
    'oil_x_loc', 'gdp_x_loc', 'oil_x_distance'
]

# Remove any missing features
features = [f for f in features if f in train.columns and f in test.columns]

print(f"   ✅ Using {len(features)} focused features")

X = train[features]
y = train[TARGET]
X_test = test[features]

# =========================
# OPTIMIZED ENSEMBLE
# =========================
print("\n[TRAINING] Optimized ensemble...")

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

oof_lgb = np.zeros(len(train))
oof_cat = np.zeros(len(train))
test_preds_lgb = np.zeros(len(test))
test_preds_cat = np.zeros(len(test))

# LightGBM with optimized parameters
print("\n   🚀 LightGBM training...")
for fold, (tr_idx, va_idx) in enumerate(kf.split(X)):
    print(f"      Fold {fold+1}/{N_FOLDS}", end=" ")
    
    Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
    ytr, yva = y.iloc[tr_idx], y.iloc[va_idx]
    
    model = lgb.LGBMRegressor(
        n_estimators=2500,
        learning_rate=0.04,
        max_depth=9,
        num_leaves=50,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=0.05,
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
    test_preds_lgb += model.predict(X_test) / N_FOLDS
    
    fold_mae = mean_absolute_error(yva, pred)
    print(f"MAE: {fold_mae:.4f}")

# CatBoost with categorical features
print("\n   🚀 CatBoost training...")

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
        iterations=1500,
        learning_rate=0.04,
        depth=9,
        l2_leaf_reg=2,
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
    test_preds_cat += model.predict(X_test_cat) / N_FOLDS
    
    fold_mae = mean_absolute_error(yva, pred)
    print(f"MAE: {fold_mae:.4f}")

cv_lgb = mean_absolute_error(y, oof_lgb)
cv_cat = mean_absolute_error(y, oof_cat)

print(f"\n   🏆 LightGBM CV MAE: {cv_lgb:.4f}")
print(f"   🏆 CatBoost CV MAE: {cv_cat:.4f}")

# Find optimal ensemble weights
best_mae = float('inf')
best_weights = None

for w_lgb in np.arange(0, 1.05, 0.05):
    w_cat = 1 - w_lgb
    oof_ens = w_lgb * oof_lgb + w_cat * oof_cat
    mae = mean_absolute_error(y, oof_ens)
    
    if mae < best_mae:
        best_mae = mae
        best_weights = (w_lgb, w_cat)

print(f"\n   🎯 Optimal weights: {best_weights[0]:.2f} LGB + {best_weights[1]:.2f} CAT")
print(f"   🏆 Ensemble CV MAE: {best_mae:.4f}")

# =========================
# GENERATE SUBMISSION
# =========================
print("\n[SUBMISSION] Generating focused predictions...")

# Clip and ensemble
test_preds_lgb = np.clip(test_preds_lgb, 50, 6000)
test_preds_cat = np.clip(test_preds_cat, 50, 6000)
test_preds_ens = best_weights[0] * test_preds_lgb + best_weights[1] * test_preds_cat
test_preds_ens = np.clip(test_preds_ens, 50, 6000)

# Create submission
test_df = pd.read_csv(TEST_CSV)
submission = pd.DataFrame({
    "data_id": test_df["data_id"], 
    TARGET: test_preds_ens
})

submission.to_csv("submission_focused_external.csv", index=False)

print(f"\n✅ Saved: submission_focused_external.csv")
print(f"   🎯 CV MAE: {best_mae:.4f}")
print(f"   🎯 TARGET: Beat 0.2317 (current best)")

if best_mae < 85:
    print(f"   🚀 EXPECTED LB: 0.20-0.22 (GOOD)")
elif best_mae < 90:
    print(f"   ⚠️  EXPECTED LB: 0.22-0.24 (MARGINAL)")
else:
    print(f"   ❌ EXPECTED LB: >0.24 (WORSE)")

print("\n" + "="*80)
print("🎯 FOCUSED MODEL COMPLETE")
print("="*80)