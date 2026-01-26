"""
FAST CV < 85 MODEL
Strategy: Optimized parameters for speed + performance
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
print("FAST CV < 85 MODEL")
print("="*80)

# =========================
# LOAD DATA + FEATURES
# =========================
print("\n[LOADING] Data and features...")
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

# Location features (PROVEN + ENHANCED)
train_sorted = train.sort_values(["geolocation_name", "year", "quarter"])

# Enhanced location statistics
stats = train.groupby("geolocation_name")[TARGET].agg([
    "mean", "std", "min", "max", "median", "count",
    ("q25", lambda x: x.quantile(0.25)),
    ("q75", lambda x: x.quantile(0.75))
])

grp = train_sorted.groupby("geolocation_name")
loc_last = grp[TARGET].last()
loc_first = grp[TARGET].first()
loc_second_last = grp[TARGET].nth(-2)

# Enhanced time features
time_first = grp["year"].first() * 4 + grp["quarter"].first()
time_last = grp["year"].last() * 4 + grp["quarter"].last()
loc_slope = (loc_last - loc_first) / (time_last - time_first).replace(0, 1)
loc_recent_slope = (loc_last - loc_second_last).fillna(0)
loc_volatility = stats["std"] / stats["mean"]
loc_range = stats["max"] - stats["min"]

# Map enhanced location features
for col in ["mean", "std", "min", "max", "median", "count", "q25", "q75"]:
    train[f"loc_{col}"] = train["geolocation_name"].map(stats[col])
    test[f"loc_{col}"] = test["geolocation_name"].map(stats[col])

for name, series in [("last", loc_last), ("first", loc_first), ("slope", loc_slope), 
                     ("recent_slope", loc_recent_slope), ("volatility", loc_volatility), ("range", loc_range)]:
    train[f"loc_{name}"] = train["geolocation_name"].map(series)
    test[f"loc_{name}"] = test["geolocation_name"].map(series)

# Country-level features
country_stats = train.groupby("country")[TARGET].agg(["mean", "std", "count"])
for col in ["mean", "std", "count"]:
    train[f"country_{col}"] = train["country"].map(country_stats[col])
    test[f"country_{col}"] = test["country"].map(country_stats[col])

# Time-location interactions
quarter_loc_stats = train.groupby(["geolocation_name", "quarter"])[TARGET].mean()
year_loc_stats = train.groupby(["geolocation_name", "year"])[TARGET].mean()

train["quarter_loc_mean"] = train.apply(lambda x: quarter_loc_stats.get((x["geolocation_name"], x["quarter"]), train[TARGET].mean()), axis=1)
test["quarter_loc_mean"] = test.apply(lambda x: quarter_loc_stats.get((x["geolocation_name"], x["quarter"]), train[TARGET].mean()), axis=1)

train["year_loc_mean"] = train.apply(lambda x: year_loc_stats.get((x["geolocation_name"], x["year"]), train[TARGET].mean()), axis=1)
test["year_loc_mean"] = test.apply(lambda x: year_loc_stats.get((x["geolocation_name"], x["year"]), train[TARGET].mean()), axis=1)

# Convert categorical
categorical_cols = ['developed_country', 'landlocked', 'access_to_airport', 'access_to_port', 
                   'access_to_highway', 'access_to_railway']

for col in categorical_cols:
    if col in train.columns and train[col].dtype == 'object':
        train[col] = (train[col] == 'Yes').astype(int)
        test[col] = (test[col] == 'Yes').astype(int)

# Enhanced features
features = [
    'year', 'quarter', 'quarter_sin', 'quarter_cos',
    'deflated_gdp_usd', 'us_cpi', 'straight_distance_to_capital_km',
    'developed_country', 'landlocked',
    'access_to_airport', 'access_to_port', 'access_to_highway', 'access_to_railway',
    # Enhanced location features
    'loc_mean', 'loc_std', 'loc_min', 'loc_max', 'loc_median', 'loc_count',
    'loc_q25', 'loc_q75', 'loc_last', 'loc_first', 'loc_slope', 'loc_recent_slope',
    'loc_volatility', 'loc_range',
    # Country features
    'country_mean', 'country_std', 'country_count',
    # Time-location interactions
    'quarter_loc_mean', 'year_loc_mean'
]

# Add key interactions
train['loc_mean_x_last'] = train['loc_mean'] * train['loc_last']
train['distance_x_loc'] = train['straight_distance_to_capital_km'] * train['loc_mean']
train['gdp_x_loc'] = train['deflated_gdp_usd'] * train['loc_mean']
train['volatility_x_range'] = train['loc_volatility'] * train['loc_range']

test['loc_mean_x_last'] = test['loc_mean'] * test['loc_last']
test['distance_x_loc'] = test['straight_distance_to_capital_km'] * test['loc_mean']
test['gdp_x_loc'] = test['deflated_gdp_usd'] * test['loc_mean']
test['volatility_x_range'] = test['loc_volatility'] * test['loc_range']

features.extend(['loc_mean_x_last', 'distance_x_loc', 'gdp_x_loc', 'volatility_x_range'])

features = [f for f in features if f in train.columns and f in test.columns]

X = train[features]
y = train[TARGET]
X_test = test[features]

print(f"   Using {len(features)} features")

# =========================
# FAST TRAINING
# =========================
print("\n[TRAINING] Fast ensemble...")

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

oof_lgb = np.zeros(len(train))
oof_cat = np.zeros(len(train))
test_preds_lgb = np.zeros(len(test))
test_preds_cat = np.zeros(len(test))

# FAST LightGBM
print("\n   LightGBM (fast + aggressive)...")
for fold, (tr_idx, va_idx) in enumerate(kf.split(X)):
    print(f"      Fold {fold+1}/{N_FOLDS}", end=" ")
    
    Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
    ytr, yva = y.iloc[tr_idx], y.iloc[va_idx]
    
    model = lgb.LGBMRegressor(
        n_estimators=4000,
        learning_rate=0.015,
        max_depth=14,
        num_leaves=150,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=0.005,
        reg_lambda=0.005,
        min_child_samples=5,
        random_state=SEED + fold,
        verbose=-1
    )
    
    model.fit(
        Xtr, ytr,
        eval_set=[(Xva, yva)],
        callbacks=[lgb.early_stopping(300, verbose=False)]
    )
    
    pred = model.predict(Xva)
    oof_lgb[va_idx] = pred
    test_preds_lgb += model.predict(X_test) / N_FOLDS
    
    fold_mae = mean_absolute_error(yva, pred)
    print(f"MAE: {fold_mae:.4f}")

# FAST CatBoost
print("\n   CatBoost (fast + optimized)...")

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
        depth=12,
        l2_leaf_reg=0.5,
        random_seed=SEED + fold,
        verbose=0,
        early_stopping_rounds=150
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
if best_mae < 85:
    print(f"\n   SUCCESS: CV < 85 achieved! ({best_mae:.4f})")
    status = "EXCELLENT"
    expected_lb = "0.19-0.22"
elif best_mae < 90:
    print(f"\n   GOOD: CV < 90 achieved ({best_mae:.4f})")
    status = "GOOD"
    expected_lb = "0.22-0.24"
else:
    print(f"\n   FAILED: CV >= 90 ({best_mae:.4f})")
    status = "BAD"
    expected_lb = ">0.24"

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

submission.to_csv("submission_fast_cv85.csv", index=False)

print(f"\nSaved: submission_fast_cv85.csv")
print(f"   CV MAE: {best_mae:.4f}")
print(f"   Expected LB: {expected_lb}")
print(f"   Status: {status}")

print("\n" + "="*80)
print("FAST CV < 85 MODEL COMPLETE")
print("="*80)