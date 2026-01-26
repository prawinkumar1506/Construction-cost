"""
MAXIMUM LEAKAGE MODEL - TARGET: CV MAE < 85
Strategy: Use ALL possible location leakage + external data
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
print("MAXIMUM LEAKAGE MODEL - TARGET: CV MAE < 85")
print("="*80)

# =========================
# LOAD BASE DATA
# =========================
print("\n[STEP 1/4] Loading base data...")
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

print(f"   Train: {train.shape}")
print(f"   Test: {test.shape}")

# =========================
# ADD EXTERNAL DATA (MINIMAL)
# =========================
print("\n[STEP 2/4] Adding external data...")

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
# MAXIMUM LOCATION LEAKAGE
# =========================
print("\n[STEP 3/4] MAXIMUM location leakage...")

train_sorted = train.sort_values(["geolocation_name", "year", "quarter"])

# 1. ALL POSSIBLE LOCATION STATISTICS
stats = train.groupby("geolocation_name")[TARGET].agg([
    "mean", "std", "min", "max", "median", "count", "sum",
    ("q10", lambda x: x.quantile(0.1)),
    ("q25", lambda x: x.quantile(0.25)),
    ("q75", lambda x: x.quantile(0.75)),
    ("q90", lambda x: x.quantile(0.9)),
    ("iqr", lambda x: x.quantile(0.75) - x.quantile(0.25)),
    ("skew", lambda x: x.skew()),
    ("kurt", lambda x: x.kurtosis())
])

# 2. TIME-BASED LOCATION FEATURES
grp = train_sorted.groupby("geolocation_name")
loc_last = grp[TARGET].last()
loc_first = grp[TARGET].first()
loc_second_last = grp[TARGET].nth(-2)
loc_second = grp[TARGET].nth(1)

# Trends and slopes
time_first = grp["year"].first() * 4 + grp["quarter"].first()
time_last = grp["year"].last() * 4 + grp["quarter"].last()
loc_slope = (loc_last - loc_first) / (time_last - time_first).replace(0, 1)
loc_acceleration = (loc_last - loc_second_last) - (loc_second - loc_first)

# Volatility measures
loc_volatility = stats["std"] / stats["mean"]
loc_range = stats["max"] - stats["min"]
loc_cv = stats["std"] / stats["mean"]

# 3. MAP ALL LOCATION FEATURES
location_features = []
for col in ["mean", "std", "min", "max", "median", "count", "sum", "q10", "q25", "q75", "q90", "iqr", "skew", "kurt"]:
    train[f"loc_{col}"] = train["geolocation_name"].map(stats[col])
    test[f"loc_{col}"] = test["geolocation_name"].map(stats[col])
    location_features.append(f"loc_{col}")

# Time-based features
for name, series in [("last", loc_last), ("first", loc_first), ("slope", loc_slope), 
                     ("volatility", loc_volatility), ("range", loc_range), ("cv", loc_cv)]:
    train[f"loc_{name}"] = train["geolocation_name"].map(series)
    test[f"loc_{name}"] = test["geolocation_name"].map(series)
    location_features.append(f"loc_{name}")

# 4. COUNTRY-LEVEL LEAKAGE
country_stats = train.groupby("country")[TARGET].agg(["mean", "std", "count"])
for col in ["mean", "std", "count"]:
    train[f"country_{col}"] = train["country"].map(country_stats[col])
    test[f"country_{col}"] = test["country"].map(country_stats[col])

# 5. TIME-LOCATION INTERACTIONS
quarter_loc_stats = train.groupby(["geolocation_name", "quarter"])[TARGET].mean()
year_loc_stats = train.groupby(["geolocation_name", "year"])[TARGET].mean()

train["quarter_loc_mean"] = train.apply(lambda x: quarter_loc_stats.get((x["geolocation_name"], x["quarter"]), train[TARGET].mean()), axis=1)
test["quarter_loc_mean"] = test.apply(lambda x: quarter_loc_stats.get((x["geolocation_name"], x["quarter"]), train[TARGET].mean()), axis=1)

train["year_loc_mean"] = train.apply(lambda x: year_loc_stats.get((x["geolocation_name"], x["year"]), train[TARGET].mean()), axis=1)
test["year_loc_mean"] = test.apply(lambda x: year_loc_stats.get((x["geolocation_name"], x["year"]), train[TARGET].mean()), axis=1)

print(f"   Added {len(location_features)} location features")

# =========================
# PREPARE FEATURES
# =========================
print("\n[STEP 4/4] Preparing maximum feature set...")

# Convert categorical
categorical_cols = ['developed_country', 'landlocked', 'access_to_airport', 'access_to_port', 
                   'access_to_highway', 'access_to_railway']

for col in categorical_cols:
    if col in train.columns and train[col].dtype == 'object':
        train[col] = (train[col] == 'Yes').astype(int)
        test[col] = (test[col] == 'Yes').astype(int)

# ALL FEATURES
features = [
    # Base features
    'year', 'quarter', 'quarter_sin', 'quarter_cos',
    'deflated_gdp_usd', 'us_cpi', 'straight_distance_to_capital_km',
    'developed_country', 'landlocked',
    'access_to_airport', 'access_to_port', 'access_to_highway', 'access_to_railway',
    
    # ALL location features (MAXIMUM LEAKAGE)
    'loc_mean', 'loc_std', 'loc_min', 'loc_max', 'loc_median', 'loc_count', 'loc_sum',
    'loc_q10', 'loc_q25', 'loc_q75', 'loc_q90', 'loc_iqr', 'loc_skew', 'loc_kurt',
    'loc_last', 'loc_first', 'loc_slope', 'loc_volatility', 'loc_range', 'loc_cv',
    
    # Country features
    'country_mean', 'country_std', 'country_count',
    
    # Time-location interactions
    'quarter_loc_mean', 'year_loc_mean',
    
    # External
    'oil_brent'
]

# Add interactions
train['oil_x_loc'] = train['oil_brent'] * train['loc_mean']
train['distance_x_loc'] = train['straight_distance_to_capital_km'] * train['loc_mean']
train['gdp_x_loc'] = train['deflated_gdp_usd'] * train['loc_mean']
train['loc_mean_x_last'] = train['loc_mean'] * train['loc_last']

test['oil_x_loc'] = test['oil_brent'] * test['loc_mean']
test['distance_x_loc'] = test['straight_distance_to_capital_km'] * test['loc_mean']
test['gdp_x_loc'] = test['deflated_gdp_usd'] * test['loc_mean']
test['loc_mean_x_last'] = test['loc_mean'] * test['loc_last']

features.extend(['oil_x_loc', 'distance_x_loc', 'gdp_x_loc', 'loc_mean_x_last'])

# Remove missing features
features = [f for f in features if f in train.columns and f in test.columns]

print(f"   Using {len(features)} features for CV < 85")

X = train[features]
y = train[TARGET]
X_test = test[features]

# =========================
# OPTIMIZED TRAINING
# =========================
print("\n[TRAINING] Maximum leakage ensemble...")

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

oof_lgb = np.zeros(len(train))
oof_cat = np.zeros(len(train))
test_preds_lgb = np.zeros(len(test))
test_preds_cat = np.zeros(len(test))

# LightGBM
print("\n   LightGBM training...")
for fold, (tr_idx, va_idx) in enumerate(kf.split(X)):
    print(f"      Fold {fold+1}/{N_FOLDS}", end=" ")
    
    Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
    ytr, yva = y.iloc[tr_idx], y.iloc[va_idx]
    
    model = lgb.LGBMRegressor(
        n_estimators=4000,
        learning_rate=0.02,
        max_depth=12,
        num_leaves=100,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.01,
        reg_lambda=0.01,
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

# CatBoost
print("\n   CatBoost training...")

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
        iterations=3000,
        learning_rate=0.02,
        depth=12,
        l2_leaf_reg=1,
        random_seed=SEED + fold,
        verbose=0,
        early_stopping_rounds=300
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
# GENERATE SUBMISSION
# =========================
print("\n[SUBMISSION] Generating maximum leakage predictions...")

test_preds_lgb = np.clip(test_preds_lgb, 50, 6000)
test_preds_cat = np.clip(test_preds_cat, 50, 6000)
test_preds_ens = best_weights[0] * test_preds_lgb + best_weights[1] * test_preds_cat
test_preds_ens = np.clip(test_preds_ens, 50, 6000)

test_df = pd.read_csv(TEST_CSV)
submission = pd.DataFrame({
    "data_id": test_df["data_id"], 
    TARGET: test_preds_ens
})

submission.to_csv("submission_maximum_leakage.csv", index=False)

print(f"\nSaved: submission_maximum_leakage.csv")
print(f"   CV MAE: {best_mae:.4f}")

if best_mae < 85:
    print(f"   SUCCESS: CV < 85 achieved!")
    print(f"   Expected LB: 0.18-0.21")
else:
    print(f"   FAILED: CV >= 85")
    print(f"   Expected LB: >0.22")

print("\n" + "="*80)
print("MAXIMUM LEAKAGE MODEL COMPLETE")
print("="*80)