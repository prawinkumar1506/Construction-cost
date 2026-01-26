"""
BREAKTHROUGH TO 0.18 - FIXED VERSION
Multi-seed ensemble + All external data + Advanced features
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
N_FOLDS = 7
SEEDS = [42, 123, 456, 789, 999]

print("="*80)
print("BREAKTHROUGH TO 0.18 - FIXED VERSION")
print("="*80)

# =========================
# LOAD DATA
# =========================
train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)

def add_time_features(df):
    df = df.copy()
    df["year"] = df["quarter_label"].str[:4].astype(int)
    df["quarter"] = df["quarter_label"].str[-1].astype(int)
    df["quarter_sin"] = np.sin(2 * np.pi * df["quarter"] / 4)
    df["quarter_cos"] = np.cos(2 * np.pi * df["quarter"] / 4)
    df["year_sin"] = np.sin(2 * np.pi * (df["year"] - 2014) / 10)
    df["year_cos"] = np.cos(2 * np.pi * (df["year"] - 2014) / 10)
    return df

train = add_time_features(train)
test = add_time_features(test)

# =========================
# ALL EXTERNAL DATA
# =========================
print("\n[EXTERNAL] Loading ALL external data...")

# Oil prices with volatility
try:
    comm_raw = pd.read_csv("external/commodity_prices.csv")
    comm_raw['date'] = pd.to_datetime(comm_raw['date'])
    comm_raw['year'] = comm_raw['date'].dt.year
    
    oil_yearly = comm_raw.groupby('year').agg({
        'oil_brent': ['mean', 'std', 'min', 'max'],
        'oil_dubai': ['mean', 'std'],
        'sugar_world': 'mean'
    }).reset_index()
    oil_yearly.columns = ['year', 'oil_brent_mean', 'oil_brent_std', 'oil_brent_min', 'oil_brent_max',
                         'oil_dubai_mean', 'oil_dubai_std', 'sugar_mean']
    oil_yearly['year'] = oil_yearly['year'].astype(int)
    oil_yearly['oil_volatility'] = oil_yearly['oil_brent_std'] / oil_yearly['oil_brent_mean']
    
    train = train.merge(oil_yearly, on="year", how="left")
    test = test.merge(oil_yearly, on="year", how="left")
    
    for col in oil_yearly.columns[1:]:
        train[col] = train[col].fillna(train[col].mean())
        test[col] = test[col].fillna(train[col].mean())
    
    print(f"   Added oil data with volatility")
except:
    for col in ['oil_brent_mean', 'oil_dubai_mean', 'sugar_mean', 'oil_volatility']:
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

# Labour participation
try:
    labour_raw = pd.read_csv("external/EAP_2WAP_SEX_AGE_RT_A-20260126T1053.csv", low_memory=False)
    labour_clean = labour_raw[(labour_raw['sex.label'] == 'Total') & 
                             (labour_raw['classif1.label'].str.contains('15+', na=False))].copy()
    labour_clean = labour_clean[['ref_area.label', 'time', 'obs_value']].copy()
    labour_clean.columns = ['country', 'year', 'labour_rate']
    labour_clean['year'] = labour_clean['year'].astype(int)
    labour_clean = labour_clean.drop_duplicates(['country', 'year']).dropna()
    
    train = train.merge(labour_clean, on=["country", "year"], how="left")
    test = test.merge(labour_clean, on=["country", "year"], how="left")
    
    train['labour_rate'] = train['labour_rate'].fillna(60)
    test['labour_rate'] = test['labour_rate'].fillna(60)
    
    print(f"   Added labour data")
except:
    train['labour_rate'] = 60.0
    test['labour_rate'] = 60.0

# =========================
# ECONOMIC GROWTH FEATURES
# =========================
print("\n[GROWTH] Economic growth features...")

train_sorted = train.sort_values(["country", "year", "quarter"])
test_sorted = test.sort_values(["country", "year", "quarter"])

# Oil price changes
train["oil_growth_1q"] = train_sorted.groupby("country")["oil_brent_mean"].pct_change(1).fillna(0).values
train["oil_growth_4q"] = train_sorted.groupby("country")["oil_brent_mean"].pct_change(4).fillna(0).values
test["oil_growth_1q"] = test_sorted.groupby("country")["oil_brent_mean"].pct_change(1).fillna(0).values
test["oil_growth_4q"] = test_sorted.groupby("country")["oil_brent_mean"].pct_change(4).fillna(0).values

# GDP changes
train["gdp_growth_1q"] = train_sorted.groupby("country")["gdp_deflator"].pct_change(1).fillna(0).values
train["gdp_growth_4q"] = train_sorted.groupby("country")["gdp_deflator"].pct_change(4).fillna(0).values
test["gdp_growth_1q"] = test_sorted.groupby("country")["gdp_deflator"].pct_change(1).fillna(0).values
test["gdp_growth_4q"] = test_sorted.groupby("country")["gdp_deflator"].pct_change(4).fillna(0).values

# Labour changes
train["labour_change_1q"] = train_sorted.groupby("country")["labour_rate"].diff(1).fillna(0).values
train["labour_change_4q"] = train_sorted.groupby("country")["labour_rate"].diff(4).fillna(0).values
test["labour_change_1q"] = test_sorted.groupby("country")["labour_rate"].diff(1).fillna(0).values
test["labour_change_4q"] = test_sorted.groupby("country")["labour_rate"].diff(4).fillna(0).values

print(f"   Added growth features")

# =========================
# ADVANCED LOCATION FEATURES
# =========================
print("\n[LOCATION] Advanced location features...")

train_sorted = train.sort_values(["geolocation_name", "year", "quarter"])

# Enhanced statistics
stats = train.groupby("geolocation_name")[TARGET].agg([
    "mean", "std", "min", "max", "median", "count",
    ("q10", lambda x: x.quantile(0.1)),
    ("q25", lambda x: x.quantile(0.25)),
    ("q75", lambda x: x.quantile(0.75)),
    ("q90", lambda x: x.quantile(0.9)),
    ("skew", lambda x: x.skew()),
    ("kurt", lambda x: x.kurtosis())
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

# Map all location features
for col in ["mean", "std", "min", "max", "median", "count", "q10", "q25", "q75", "q90", "skew", "kurt"]:
    train[f"loc_{col}"] = train["geolocation_name"].map(stats[col])
    test[f"loc_{col}"] = test["geolocation_name"].map(stats[col])

for name, series in [("last", loc_last), ("first", loc_first), ("slope", loc_slope), 
                     ("recent_trend", loc_recent_trend), ("acceleration", loc_acceleration),
                     ("volatility", loc_volatility)]:
    train[f"loc_{name}"] = train["geolocation_name"].map(series)
    test[f"loc_{name}"] = test["geolocation_name"].map(series)

# Country features
country_stats = train.groupby("country")[TARGET].agg(["mean", "std", "count", "min", "max"])
for col in ["mean", "std", "count", "min", "max"]:
    train[f"country_{col}"] = train["country"].map(country_stats[col])
    test[f"country_{col}"] = test["country"].map(country_stats[col])

print(f"   Added advanced location features")

# =========================
# CONVERT CATEGORICAL
# =========================
categorical_cols = ['developed_country', 'landlocked', 'access_to_airport', 'access_to_port', 
                   'access_to_highway', 'access_to_railway']

for col in categorical_cols:
    if col in train.columns and train[col].dtype == 'object':
        train[col] = (train[col] == 'Yes').astype(int)
        test[col] = (test[col] == 'Yes').astype(int)

# =========================
# POWER INTERACTIONS
# =========================
print("\n[INTERACTIONS] Power interactions...")

# Economic interactions
train['oil_gdp_power'] = train['oil_brent_mean'] * train['gdp_deflator'] / 100
train['labour_gdp_power'] = train['labour_rate'] * train['gdp_deflator'] / 100
train['oil_labour_power'] = train['oil_brent_mean'] * train['labour_rate'] / 100

test['oil_gdp_power'] = test['oil_brent_mean'] * test['gdp_deflator'] / 100
test['labour_gdp_power'] = test['labour_rate'] * test['gdp_deflator'] / 100
test['oil_labour_power'] = test['oil_brent_mean'] * test['labour_rate'] / 100

# Location-economic interactions
train['loc_oil_power'] = train['loc_mean'] * train['oil_brent_mean'] / 100
train['loc_gdp_power'] = train['loc_mean'] * train['gdp_deflator'] / 100
train['loc_distance_power'] = train['loc_mean'] * train['straight_distance_to_capital_km'] / 1000

test['loc_oil_power'] = test['loc_mean'] * test['oil_brent_mean'] / 100
test['loc_gdp_power'] = test['loc_mean'] * test['gdp_deflator'] / 100
test['loc_distance_power'] = test['loc_mean'] * test['straight_distance_to_capital_km'] / 1000

# Growth momentum
train['growth_momentum'] = train['oil_growth_4q'] * train['gdp_growth_4q'] * train['labour_change_4q']
test['growth_momentum'] = test['oil_growth_4q'] * test['gdp_growth_4q'] * test['labour_change_4q']

print(f"   Added power interactions")

# =========================
# ULTIMATE FEATURE SET
# =========================
features = [
    # Base features
    'year', 'quarter', 'quarter_sin', 'quarter_cos', 'year_sin', 'year_cos',
    'deflated_gdp_usd', 'us_cpi', 'straight_distance_to_capital_km',
    'developed_country', 'landlocked',
    'access_to_airport', 'access_to_port', 'access_to_highway', 'access_to_railway',
    
    # Advanced location features
    'loc_mean', 'loc_std', 'loc_min', 'loc_max', 'loc_median', 'loc_count',
    'loc_q10', 'loc_q25', 'loc_q75', 'loc_q90', 'loc_skew', 'loc_kurt',
    'loc_last', 'loc_first', 'loc_slope', 'loc_recent_trend', 'loc_acceleration', 'loc_volatility',
    
    # Country features
    'country_mean', 'country_std', 'country_count', 'country_min', 'country_max',
    
    # External features
    'oil_brent_mean', 'oil_dubai_mean', 'oil_volatility', 'gdp_deflator', 'labour_rate',
    
    # Growth features
    'oil_growth_1q', 'oil_growth_4q', 'gdp_growth_1q', 'gdp_growth_4q',
    'labour_change_1q', 'labour_change_4q',
    
    # Power interactions
    'oil_gdp_power', 'labour_gdp_power', 'oil_labour_power',
    'loc_oil_power', 'loc_gdp_power', 'loc_distance_power', 'growth_momentum'
]

features = [f for f in features if f in train.columns and f in test.columns]

X = train[features].fillna(0)
y = train[TARGET]
X_test = test[features].fillna(0)

print(f"   Using {len(features)} ultimate features")

# =========================
# MULTI-SEED ENSEMBLE
# =========================
print("\n[TRAINING] Multi-seed ultimate ensemble...")

all_oof = []
all_test_preds = []

for seed_idx, seed in enumerate(SEEDS):
    print(f"\n   SEED {seed} ({seed_idx+1}/{len(SEEDS)})")
    
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    
    oof_lgb = np.zeros(len(train))
    oof_cat = np.zeros(len(train))
    test_preds_lgb = np.zeros(len(test))
    test_preds_cat = np.zeros(len(test))
    
    # LightGBM
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X)):
        Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
        ytr, yva = y.iloc[tr_idx], y.iloc[va_idx]
        
        model = lgb.LGBMRegressor(
            n_estimators=5000,
            learning_rate=0.015,
            max_depth=12,
            num_leaves=120,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=0.005,
            reg_lambda=0.005,
            min_child_samples=3,
            random_state=seed + fold,
            verbose=-1
        )
        
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)], 
                 callbacks=[lgb.early_stopping(400, verbose=False)])
        
        oof_lgb[va_idx] = model.predict(Xva)
        test_preds_lgb += model.predict(X_test) / N_FOLDS
    
    # CatBoost
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
            iterations=4000,
            learning_rate=0.015,
            depth=12,
            l2_leaf_reg=0.5,
            random_seed=seed + fold,
            verbose=0,
            early_stopping_rounds=400
        )
        
        model.fit(Xtr, ytr, cat_features=cat_indices, 
                 eval_set=(Xva, yva), verbose=False)
        
        oof_cat[va_idx] = model.predict(Xva)
        test_preds_cat += model.predict(X_test_cat) / N_FOLDS
    
    # Optimal blend for this seed
    best_mae = float('inf')
    best_w = 0.5
    
    for w in np.arange(0, 1.05, 0.05):
        oof_blend = w * oof_lgb + (1-w) * oof_cat
        mae = mean_absolute_error(y, oof_blend)
        if mae < best_mae:
            best_mae = mae
            best_w = w
    
    oof_final = best_w * oof_lgb + (1-best_w) * oof_cat
    test_final = best_w * test_preds_lgb + (1-best_w) * test_preds_cat
    
    all_oof.append(oof_final)
    all_test_preds.append(test_final)
    
    print(f"      Seed {seed} CV MAE: {best_mae:.4f}")

# Final ensemble across seeds
final_oof = np.mean(all_oof, axis=0)
final_test_preds = np.mean(all_test_preds, axis=0)

final_cv = mean_absolute_error(y, final_oof)
print(f"\n   FINAL MULTI-SEED CV MAE: {final_cv:.4f}")

# =========================
# GENERATE SUBMISSION
# =========================
final_test_preds = np.clip(final_test_preds, 50, 6000)

test_df = pd.read_csv(TEST_CSV)
submission = pd.DataFrame({
    "data_id": test_df["data_id"], 
    TARGET: final_test_preds
})

submission.to_csv("submission_breakthrough_018_fixed.csv", index=False)

print(f"\nSaved: submission_breakthrough_018_fixed.csv")
print(f"   CV MAE: {final_cv:.4f}")

if final_cv < 75:
    print(f"   BREAKTHROUGH: Expected LB 0.17-0.19 - TARGET 0.18 ACHIEVED!")
elif final_cv < 80:
    print(f"   EXCELLENT: Expected LB 0.18-0.20 - VERY CLOSE TO 0.18!")
elif final_cv < 85:
    print(f"   GOOD: Expected LB 0.19-0.21 - PROGRESS TOWARD 0.18!")
else:
    print(f"   PROGRESS: Expected LB 0.20-0.23 - KEEP IMPROVING!")

print("\n" + "="*80)
print("BREAKTHROUGH MODEL COMPLETE - TARGETING 0.18!")
print("="*80)