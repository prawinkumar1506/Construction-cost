"""
FAST BREAKTHROUGH TO 0.18
Optimized for speed while maintaining performance
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import time
import warnings
warnings.filterwarnings("ignore")

TARGET = "construction_cost_per_m2_usd"
TRAIN_CSV = "datasets/train_tabular.csv"
TEST_CSV = "datasets/evaluation_tabular_no_target.csv"
N_FOLDS = 5  # Reduced
SEEDS = [42, 123, 456]  # Reduced to 3 seeds

print("="*80)
print("FAST BREAKTHROUGH TO 0.18")
print("="*80)

start_time = time.time()

# =========================
# LOAD DATA
# =========================
print("\n[1/6] Loading data...")
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
# EXTERNAL DATA (FAST)
# =========================
print("\n[2/6] Loading external data...")

# Oil prices
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
    
    print(f"   ✓ Oil data loaded")
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
    
    print(f"   ✓ GDP deflator loaded")
except:
    train['gdp_deflator'] = 100.0
    test['gdp_deflator'] = 100.0

# =========================
# LOCATION FEATURES (ENHANCED)
# =========================
print("\n[3/6] Creating enhanced location features...")

train_sorted = train.sort_values(["geolocation_name", "year", "quarter"])

# Enhanced location statistics
stats = train.groupby("geolocation_name")[TARGET].agg([
    "mean", "std", "min", "max", "median", "count",
    ("q10", lambda x: x.quantile(0.1)),
    ("q25", lambda x: x.quantile(0.25)),
    ("q75", lambda x: x.quantile(0.75)),
    ("q90", lambda x: x.quantile(0.9)),
    ("skew", lambda x: x.skew()),
    ("iqr", lambda x: x.quantile(0.75) - x.quantile(0.25))
])

grp = train_sorted.groupby("geolocation_name")
loc_last = grp[TARGET].last()
loc_first = grp[TARGET].first()
loc_second_last = grp[TARGET].nth(-2)
loc_second = grp[TARGET].nth(1)

# Advanced time features
time_first = grp["year"].first() * 4 + grp["quarter"].first()
time_last = grp["year"].last() * 4 + grp["quarter"].last()
loc_slope = (loc_last - loc_first) / (time_last - time_first).replace(0, 1)
loc_recent_trend = (loc_last - loc_second_last).fillna(0)
loc_acceleration = ((loc_last - loc_second_last) - (loc_second - loc_first)).fillna(0)
loc_volatility = stats["std"] / stats["mean"]
loc_range_norm = (stats["max"] - stats["min"]) / stats["mean"]

# Map enhanced features
for col in ["mean", "std", "min", "max", "median", "count", "q10", "q25", "q75", "q90", "skew", "iqr"]:
    train[f"loc_{col}"] = train["geolocation_name"].map(stats[col])
    test[f"loc_{col}"] = test["geolocation_name"].map(stats[col])

for name, series in [("last", loc_last), ("first", loc_first), ("slope", loc_slope), 
                     ("recent_trend", loc_recent_trend), ("acceleration", loc_acceleration),
                     ("volatility", loc_volatility), ("range_norm", loc_range_norm)]:
    train[f"loc_{name}"] = train["geolocation_name"].map(series)
    test[f"loc_{name}"] = test["geolocation_name"].map(series)

# Enhanced country features
country_stats = train.groupby("country")[TARGET].agg(["mean", "std", "count", "min", "max"])
for col in ["mean", "std", "count", "min", "max"]:
    train[f"country_{col}"] = train["country"].map(country_stats[col])
    test[f"country_{col}"] = test["country"].map(country_stats[col])

# Time-location interactions
quarter_loc_stats = train.groupby(["geolocation_name", "quarter"])[TARGET].mean()
year_loc_stats = train.groupby(["geolocation_name", "year"])[TARGET].mean()

train["quarter_loc_mean"] = train.apply(lambda x: quarter_loc_stats.get((x["geolocation_name"], x["quarter"]), train[TARGET].mean()), axis=1)
test["quarter_loc_mean"] = test.apply(lambda x: quarter_loc_stats.get((x["geolocation_name"], x["quarter"]), train[TARGET].mean()), axis=1)

train["year_loc_mean"] = train.apply(lambda x: year_loc_stats.get((x["geolocation_name"], x["year"]), train[TARGET].mean()), axis=1)
test["year_loc_mean"] = test.apply(lambda x: year_loc_stats.get((x["geolocation_name"], x["year"]), train[TARGET].mean()), axis=1)

print(f"   ✓ Enhanced location features created")

# =========================
# ENHANCED GROWTH FEATURES
# =========================
print("\n[4/6] Creating enhanced growth features...")

train_sorted = train.sort_values(["country", "year", "quarter"])
test_sorted = test.sort_values(["country", "year", "quarter"])

# Oil price changes (multiple timeframes)
train["oil_growth_1q"] = train_sorted.groupby("country")["oil_brent_mean"].pct_change(1).fillna(0).values
train["oil_growth_4q"] = train_sorted.groupby("country")["oil_brent_mean"].pct_change(4).fillna(0).values
test["oil_growth_1q"] = test_sorted.groupby("country")["oil_brent_mean"].pct_change(1).fillna(0).values
test["oil_growth_4q"] = test_sorted.groupby("country")["oil_brent_mean"].pct_change(4).fillna(0).values

# GDP changes
train["gdp_growth_1q"] = train_sorted.groupby("country")["gdp_deflator"].pct_change(1).fillna(0).values
train["gdp_growth_4q"] = train_sorted.groupby("country")["gdp_deflator"].pct_change(4).fillna(0).values
test["gdp_growth_1q"] = test_sorted.groupby("country")["gdp_deflator"].pct_change(1).fillna(0).values
test["gdp_growth_4q"] = test_sorted.groupby("country")["gdp_deflator"].pct_change(4).fillna(0).values

# Economic momentum
train["economic_momentum"] = train["oil_growth_4q"] * train["gdp_growth_4q"]
test["economic_momentum"] = test["oil_growth_4q"] * test["gdp_growth_4q"]

print(f"   ✓ Enhanced growth features created")

# =========================
# CONVERT CATEGORICAL & INTERACTIONS
# =========================
print("\n[5/6] Final feature engineering...")

# Convert categorical
categorical_cols = ['developed_country', 'landlocked', 'access_to_airport', 'access_to_port', 
                   'access_to_highway', 'access_to_railway']

for col in categorical_cols:
    if col in train.columns and train[col].dtype == 'object':
        train[col] = (train[col] == 'Yes').astype(int)
        test[col] = (test[col] == 'Yes').astype(int)

# Enhanced interactions
train['oil_x_loc'] = train['oil_brent_mean'] * train['loc_mean'] / 100
train['gdp_x_loc'] = train['gdp_deflator'] * train['loc_mean'] / 100
train['distance_x_loc'] = train['straight_distance_to_capital_km'] * train['loc_mean'] / 1000
train['volatility_x_loc'] = train['oil_volatility'] * train['loc_volatility']
train['growth_x_loc'] = train['economic_momentum'] * train['loc_mean']
train['trend_x_acceleration'] = train['loc_recent_trend'] * train['loc_acceleration']
train['country_x_loc'] = train['country_mean'] * train['loc_mean'] / 1000
train['quarter_interaction'] = train['quarter_loc_mean'] * train['oil_brent_mean'] / 100

test['oil_x_loc'] = test['oil_brent_mean'] * test['loc_mean'] / 100
test['gdp_x_loc'] = test['gdp_deflator'] * test['loc_mean'] / 100
test['distance_x_loc'] = test['straight_distance_to_capital_km'] * test['loc_mean'] / 1000
test['volatility_x_loc'] = test['oil_volatility'] * test['loc_volatility']
test['growth_x_loc'] = test['economic_momentum'] * test['loc_mean']
test['trend_x_acceleration'] = test['loc_recent_trend'] * test['loc_acceleration']
test['country_x_loc'] = test['country_mean'] * test['loc_mean'] / 1000
test['quarter_interaction'] = test['quarter_loc_mean'] * test['oil_brent_mean'] / 100

# Enhanced feature set
features = [
    'year', 'quarter', 'quarter_sin', 'quarter_cos',
    'deflated_gdp_usd', 'us_cpi', 'straight_distance_to_capital_km',
    'developed_country', 'landlocked',
    'access_to_airport', 'access_to_port', 'access_to_highway', 'access_to_railway',
    # Enhanced location features
    'loc_mean', 'loc_std', 'loc_min', 'loc_max', 'loc_median', 'loc_count',
    'loc_q10', 'loc_q25', 'loc_q75', 'loc_q90', 'loc_skew', 'loc_iqr',
    'loc_last', 'loc_first', 'loc_slope', 'loc_recent_trend', 'loc_acceleration',
    'loc_volatility', 'loc_range_norm',
    # Enhanced country features
    'country_mean', 'country_std', 'country_count', 'country_min', 'country_max',
    # Time-location interactions
    'quarter_loc_mean', 'year_loc_mean',
    # External features
    'oil_brent_mean', 'oil_dubai_mean', 'oil_volatility', 'gdp_deflator',
    # Enhanced growth features
    'oil_growth_1q', 'oil_growth_4q', 'gdp_growth_1q', 'gdp_growth_4q', 'economic_momentum',
    # Enhanced interactions
    'oil_x_loc', 'gdp_x_loc', 'distance_x_loc', 'volatility_x_loc',
    'growth_x_loc', 'trend_x_acceleration', 'country_x_loc', 'quarter_interaction'
]

features = [f for f in features if f in train.columns and f in test.columns]

X = train[features].fillna(0)
y = train[TARGET]
X_test = test[features].fillna(0)

print(f"   ✓ Using {len(features)} features")

# =========================
# FAST MULTI-SEED ENSEMBLE
# =========================
print(f"\n[6/6] Training {len(SEEDS)} seeds × {N_FOLDS} folds = {len(SEEDS) * N_FOLDS} models...")

all_oof = []
all_test_preds = []
total_models = len(SEEDS) * N_FOLDS * 2  # 2 algorithms
current_model = 0

for seed_idx, seed in enumerate(SEEDS):
    print(f"\n   SEED {seed} ({seed_idx+1}/{len(SEEDS)})")
    
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    
    oof_lgb = np.zeros(len(train))
    oof_cat = np.zeros(len(train))
    test_preds_lgb = np.zeros(len(test))
    test_preds_cat = np.zeros(len(test))
    
    # LightGBM (FAST PARAMETERS)
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X)):
        current_model += 1
        print(f"      LGB Fold {fold+1}/{N_FOLDS} ({current_model}/{total_models})", end=" ")
        
        Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
        ytr, yva = y.iloc[tr_idx], y.iloc[va_idx]
        
        model = lgb.LGBMRegressor(
            n_estimators=3000,  # INCREASED
            learning_rate=0.02,  # SLOWER FOR BETTER LEARNING
            max_depth=12,  # DEEPER
            num_leaves=100,  # MORE LEAVES
            subsample=0.7,  # MORE AGGRESSIVE
            colsample_bytree=0.7,
            reg_alpha=0.005,  # LESS REGULARIZATION
            reg_lambda=0.005,
            min_child_samples=5,  # LESS REGULARIZATION
            random_state=seed + fold,
            verbose=-1
        )
        
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)], 
                 callbacks=[lgb.early_stopping(200, verbose=False)])  # MORE PATIENT
        
        oof_lgb[va_idx] = model.predict(Xva)
        test_preds_lgb += model.predict(X_test) / N_FOLDS
        
        fold_mae = mean_absolute_error(yva, model.predict(Xva))
        print(f"MAE: {fold_mae:.2f}")
    
    # CatBoost (FAST PARAMETERS)
    X_cat = X.copy()
    X_test_cat = X_test.copy()
    X_cat['country'] = train['country']
    X_cat['geolocation_name'] = train['geolocation_name']
    X_test_cat['country'] = test['country']
    X_test_cat['geolocation_name'] = test['geolocation_name']
    
    cat_indices = [X_cat.columns.get_loc('country'), X_cat.columns.get_loc('geolocation_name')]
    
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_cat)):
        current_model += 1
        print(f"      CAT Fold {fold+1}/{N_FOLDS} ({current_model}/{total_models})", end=" ")
        
        Xtr, Xva = X_cat.iloc[tr_idx], X_cat.iloc[va_idx]
        ytr, yva = y.iloc[tr_idx], y.iloc[va_idx]
        
        model = CatBoostRegressor(
            iterations=2000,  # INCREASED
            learning_rate=0.03,  # SLOWER
            depth=10,  # DEEPER
            l2_leaf_reg=0.5,  # LESS REGULARIZATION
            random_seed=seed + fold,
            verbose=0,
            early_stopping_rounds=150  # MORE PATIENT
        )
        
        model.fit(Xtr, ytr, cat_features=cat_indices, 
                 eval_set=(Xva, yva), verbose=False)
        
        oof_cat[va_idx] = model.predict(Xva)
        test_preds_cat += model.predict(X_test_cat) / N_FOLDS
        
        fold_mae = mean_absolute_error(yva, model.predict(Xva))
        print(f"MAE: {fold_mae:.2f}")
    
    # Optimal blend for this seed
    best_mae = float('inf')
    best_w = 0.5
    
    for w in np.arange(0, 1.05, 0.05):  # FINER SEARCH
        oof_blend = w * oof_lgb + (1-w) * oof_cat
        mae = mean_absolute_error(y, oof_blend)
        if mae < best_mae:
            best_mae = mae
            best_w = w
    
    oof_final = best_w * oof_lgb + (1-best_w) * oof_cat
    test_final = best_w * test_preds_lgb + (1-best_w) * test_preds_cat
    
    all_oof.append(oof_final)
    all_test_preds.append(test_final)
    
    print(f"   → Seed {seed} CV MAE: {best_mae:.4f} (Weight: {best_w:.1f} LGB)")

# Final ensemble across seeds
final_oof = np.mean(all_oof, axis=0)
final_test_preds = np.mean(all_test_preds, axis=0)

final_cv = mean_absolute_error(y, final_oof)

elapsed_time = time.time() - start_time
print(f"\n   🏆 FINAL CV MAE: {final_cv:.4f}")
print(f"   ⏱️  Total time: {elapsed_time/60:.1f} minutes")

# =========================
# GENERATE SUBMISSION
# =========================
final_test_preds = np.clip(final_test_preds, 50, 6000)

test_df = pd.read_csv(TEST_CSV)
submission = pd.DataFrame({
    "data_id": test_df["data_id"], 
    TARGET: final_test_preds
})

submission.to_csv("submission_fast_breakthrough_018.csv", index=False)

print(f"\n✅ Saved: submission_fast_breakthrough_018.csv")
print(f"   CV MAE: {final_cv:.4f}")

if final_cv < 75:
    print(f"   🚀 BREAKTHROUGH: Expected LB 0.17-0.19 - TARGET 0.18 ACHIEVED!")
elif final_cv < 80:
    print(f"   🎯 EXCELLENT: Expected LB 0.18-0.20 - VERY CLOSE TO 0.18!")
elif final_cv < 85:
    print(f"   ✅ GOOD: Expected LB 0.19-0.21 - SOLID PROGRESS!")
elif final_cv < 90:
    print(f"   📈 PROGRESS: Expected LB 0.20-0.23 - MOVING FORWARD!")
else:
    print(f"   ⚠️  BASELINE: Expected LB 0.23+ - NEEDS MORE WORK!")

print("\n" + "="*80)
print("FAST BREAKTHROUGH MODEL COMPLETE!")
print("="*80)