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
# BALANCED EXTERNAL DATA
# =========================
print("\n[EXTERNAL] Adding balanced external data...")

# Oil prices with volatility
try:
    comm_raw = pd.read_csv("external/commodity_prices.csv")
    comm_raw['date'] = pd.to_datetime(comm_raw['date'])
    comm_raw['year'] = comm_raw['date'].dt.year
    
    oil_yearly = comm_raw.groupby('year').agg({
        'oil_brent': ['mean', 'std'],
        'oil_dubai': 'mean'
    }).reset_index()
    oil_yearly.columns = ['year', 'oil_brent_mean', 'oil_brent_std', 'oil_dubai_mean']
    oil_yearly['year'] = oil_yearly['year'].astype(int)
    oil_yearly['oil_volatility'] = oil_yearly['oil_brent_std'] / oil_yearly['oil_brent_mean']
    
    train = train.merge(oil_yearly, on="year", how="left")
    test = test.merge(oil_yearly, on="year", how="left")
    
    for col in ['oil_brent_mean', 'oil_dubai_mean', 'oil_volatility']:
        train[col] = train[col].fillna(train[col].mean())
        test[col] = test[col].fillna(train[col].mean())
    
    print(f"   Added oil data with volatility")
except:
    for col in ['oil_brent_mean', 'oil_dubai_mean', 'oil_volatility']:
        train[col] = 50.0
        test[col] = 50.0

# GDP deflator
try:
    gdp_raw = pd.read_csv("external/API_NY.GDP.DEFL.ZS_DS2_en_csv_v2_4610.csv", skiprows=4)
    year_cols = [c for c in gdp_raw.columns if c.isdigit() and int(c) >= 2010]
    
    gdp_long = gdp_raw.melt(id_vars=['Country Name'], value_vars=year_cols, 
                           var_name='year', value_name='gdp_deflator')
    gdp_long.columns = ['country', 'year', 'gdp_deflator']
    gdp_long['year'] = gdp_long['year'].astype(int)
    gdp_long = gdp_long.dropna()
    
    train = train.merge(gdp_long, on=["country", "year"], how="left")
    test = test.merge(gdp_long, on=["country", "year"], how="left")
    
    train['gdp_deflator'] = train['gdp_deflator'].fillna(100)
    test['gdp_deflator'] = test['gdp_deflator'].fillna(100)
    
    print(f"   Added GDP deflator")
except:
    train['gdp_deflator'] = 100.0
    test['gdp_deflator'] = 100.0

# =========================
# BALANCED LOCATION FEATURES
# =========================
print("\n[LOCATION] Balanced location features...")

train_sorted = train.sort_values(["geolocation_name", "year", "quarter"])

# Core location statistics + key additions
stats = train.groupby("geolocation_name")[TARGET].agg([
    "mean", "std", "min", "max", "median", "count",
    ("q25", lambda x: x.quantile(0.25)),
    ("q75", lambda x: x.quantile(0.75))
])

grp = train_sorted.groupby("geolocation_name")
loc_last = grp[TARGET].last()
loc_first = grp[TARGET].first()

# Time features
time_first = grp["year"].first() * 4 + grp["quarter"].first()
time_last = grp["year"].last() * 4 + grp["quarter"].last()
loc_slope = (loc_last - loc_first) / (time_last - time_first).replace(0, 1)
loc_volatility = stats["std"] / stats["mean"]
loc_range = stats["max"] - stats["min"]

# Map location features
for col in ["mean", "std", "min", "max", "median", "count", "q25", "q75"]:
    train[f"loc_{col}"] = train["geolocation_name"].map(stats[col])
    test[f"loc_{col}"] = test["geolocation_name"].map(stats[col])

for name, series in [("last", loc_last), ("first", loc_first), ("slope", loc_slope), 
                     ("volatility", loc_volatility), ("range", loc_range)]:
    train[f"loc_{name}"] = train["geolocation_name"].map(series)
    test[f"loc_{name}"] = test["geolocation_name"].map(series)

# Country features
country_stats = train.groupby("country")[TARGET].agg(["mean", "std", "count"])
for col in ["mean", "std", "count"]:
    train[f"country_{col}"] = train["country"].map(country_stats[col])
    test[f"country_{col}"] = test["country"].map(country_stats[col])

# Economic growth features
train_sorted = train.sort_values(["country", "year", "quarter"])
test_sorted = test.sort_values(["country", "year", "quarter"])

train["oil_growth_4q"] = train_sorted.groupby("country")["oil_brent_mean"].pct_change(4).fillna(0).values
test["oil_growth_4q"] = test_sorted.groupby("country")["oil_brent_mean"].pct_change(4).fillna(0).values

train["gdp_growth_4q"] = train_sorted.groupby("country")["gdp_deflator"].pct_change(4).fillna(0).values
test["gdp_growth_4q"] = test_sorted.groupby("country")["gdp_deflator"].pct_change(4).fillna(0).values

# Convert categorical
categorical_cols = ['developed_country', 'landlocked', 'access_to_airport', 'access_to_port', 
                   'access_to_highway', 'access_to_railway']

for col in categorical_cols:
    if col in train.columns and train[col].dtype == 'object':
        train[col] = (train[col] == 'Yes').astype(int)
        test[col] = (test[col] == 'Yes').astype(int)

print(f"   Added balanced location features")

# =========================
# BALANCED FEATURE SET
# =========================
print("\n[FEATURES] Balanced feature set...")

# Balanced feature set
features = [
    'year', 'quarter', 'quarter_sin', 'quarter_cos',
    'deflated_gdp_usd', 'us_cpi', 'straight_distance_to_capital_km',
    'developed_country', 'landlocked',
    'access_to_airport', 'access_to_port', 'access_to_highway', 'access_to_railway',
    'loc_mean', 'loc_std', 'loc_min', 'loc_max', 'loc_median', 'loc_count',
    'loc_q25', 'loc_q75', 'loc_last', 'loc_first', 'loc_slope', 'loc_volatility', 'loc_range',
    'country_mean', 'country_std', 'country_count',
    'oil_brent_mean', 'oil_dubai_mean', 'oil_volatility', 'gdp_deflator',
    'oil_growth_4q', 'gdp_growth_4q'
]

# Balanced interactions
train['oil_x_loc'] = train['oil_brent_mean'] * train['loc_mean'] / 100
train['gdp_x_loc'] = train['gdp_deflator'] * train['loc_mean'] / 100
train['distance_x_oil'] = train['straight_distance_to_capital_km'] * train['oil_brent_mean'] / 1000
train['growth_momentum'] = train['oil_growth_4q'] * train['gdp_growth_4q']
train['volatility_interaction'] = train['oil_volatility'] * train['loc_volatility']

test['oil_x_loc'] = test['oil_brent_mean'] * test['loc_mean'] / 100
test['gdp_x_loc'] = test['gdp_deflator'] * test['loc_mean'] / 100
test['distance_x_oil'] = test['straight_distance_to_capital_km'] * test['oil_brent_mean'] / 1000
test['growth_momentum'] = test['oil_growth_4q'] * test['gdp_growth_4q']
test['volatility_interaction'] = test['oil_volatility'] * test['loc_volatility']

features.extend(['oil_x_loc', 'gdp_x_loc', 'distance_x_oil', 'growth_momentum', 'volatility_interaction'])

features = [f for f in features if f in train.columns and f in test.columns]

X = train[features]
y = train[TARGET]
X_test = test[features]

print(f"   Using {len(features)} balanced features")

# =========================
# BALANCED TRAINING
# =========================
print("\n[TRAINING] Balanced ensemble...")

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

oof_lgb = np.zeros(len(train))
oof_cat = np.zeros(len(train))
test_preds_lgb = np.zeros(len(test))
test_preds_cat = np.zeros(len(test))

# BALANCED LightGBM
print("\n   LightGBM (balanced)...")
for fold, (tr_idx, va_idx) in enumerate(kf.split(X)):
    print(f"      Fold {fold+1}/{N_FOLDS}", end=" ")
    
    Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
    ytr, yva = y.iloc[tr_idx], y.iloc[va_idx]
    
    model = lgb.LGBMRegressor(
        n_estimators=3000,
        learning_rate=0.025,
        max_depth=9,
        num_leaves=50,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.05,
        reg_lambda=0.05,
        min_child_samples=15,
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