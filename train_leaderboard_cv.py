"""
🎯 LEADERBOARD-ALIGNED CV STRATEGY
Time-based split per location (not spatial split)
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupKFold
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

BASE = Path('d:/Projects/Construction-cost')
DATA_DIR = BASE / 'datasets'

# Load data
train = pd.read_csv(DATA_DIR / 'train_tabular.csv')
eval_df = pd.read_csv(DATA_DIR / 'evaluation_tabular_no_target.csv')
# Preprocessing
# Fill risk default
train['tropical_cyclone_wind_risk'] = train['tropical_cyclone_wind_risk'].fillna('Low')
eval_df['tropical_cyclone_wind_risk'] = eval_df['tropical_cyclone_wind_risk'].fillna('Low')

# Binary flags -> 0/1
bool_cols = ['developed_country', 'landlocked', 'access_to_airport', 'access_to_port', 
             'access_to_highway', 'access_to_railway', 'flood_risk_class']
for col in bool_cols:
    train[col] = (train[col] == 'Yes').astype(int)
    eval_df[col] = (eval_df[col] == 'Yes').astype(int)

# Risk mappings (numeric)
risk_map = {'Very Low': 0, 'Low': 1, 'Moderate': 2, 'High': 3, 'Very High': 4}
for col in ['seismic_hazard_zone', 'tropical_cyclone_wind_risk', 'tornadoes_wind_risk']:
    train[col] = train[col].map(risk_map)
    eval_df[col] = eval_df[col].map(risk_map)

# Preserve raw categorical strings for CatBoost
train['country_str'] = train['country'].astype(str)
train['region_str'] = train['region_economic_classification'].astype(str)
train['koppen_str'] = train['koppen_climate_zone'].astype(str)

eval_df['country_str'] = eval_df['country'].astype(str)
eval_df['region_str'] = eval_df['region_economic_classification'].astype(str)
eval_df['koppen_str'] = eval_df['koppen_climate_zone'].astype(str)

# Numeric encodings for tree models (LightGBM / XGBoost)
for col, enc_col in [
    ('country', 'country_enc'),
    ('region_economic_classification', 'region_enc'),
    ('koppen_climate_zone', 'koppen_enc')
]:
    le = LabelEncoder()
    train[enc_col] = le.fit_transform(train[col].astype(str))
    # handle unseen labels in eval by mapping where possible, else -1
    mapping = {k: v for v, k in enumerate(le.classes_)}
    eval_df[enc_col] = eval_df[col].map(mapping).fillna(-1).astype(int)

# NOTE: do NOT use location identity as a feature (prevents memorization)
# Keep geolocation_name available for GroupKFold grouping only.
train['location_id'] = train['geolocation_name'].astype(str)
eval_df['location_id'] = eval_df['geolocation_name'].astype(str)

# Time features: extract quarter number and build continuous time index
# quarter_label is like '2019-Q4'
def parse_quarter(q):
    try:
        year, qpart = q.split('-Q')
        return int(year), int(qpart)
    except Exception:
        return (np.nan, np.nan)

train[['year_from_q', 'quarter']] = train['quarter_label'].apply(lambda x: pd.Series(parse_quarter(x)))
eval_df[['year_from_q', 'quarter']] = eval_df['quarter_label'].apply(lambda x: pd.Series(parse_quarter(x)))

# prefer existing numeric 'year' column if present
if 'year' in train.columns:
    train['year_num'] = train['year'].astype(int)
else:
    train['year_num'] = train['year_from_q']
if 'year' in eval_df.columns:
    eval_df['year_num'] = eval_df['year'].astype(int)
else:
    eval_df['year_num'] = eval_df['year_from_q']

# continuous time index (quarter resolution)
train['t'] = train['year_num'] * 4 + (train['quarter'] - 1)
eval_df['t'] = eval_df['year_num'] * 4 + (eval_df['quarter'] - 1)

# normalized time and seasonal features
t_min, t_max = train['t'].min(), train['t'].max()
train['t_norm'] = (train['t'] - t_min) / (t_max - t_min + 1e-9)
eval_df['t_norm'] = (eval_df['t'] - t_min) / (t_max - t_min + 1e-9)
train['sin_q'] = np.sin(2 * np.pi * train['quarter'] / 4)
train['cos_q'] = np.cos(2 * np.pi * train['quarter'] / 4)
eval_df['sin_q'] = np.sin(2 * np.pi * eval_df['quarter'] / 4)
eval_df['cos_q'] = np.cos(2 * np.pi * eval_df['quarter'] / 4)

# Features: keep necessary columns, include both encodings and raw strings for CatBoost
drop_cols = ['data_id', 'geolocation_name', 'quarter_label', 'sentinel2_tiff_file_name', 
             'viirs_tiff_file_name', 'construction_cost_per_m2_usd', 'deflated_gdp_usd']

# build feature list explicitly to control types
feature_cols = []
# numeric / engineered
num_feats = ['developed_country', 'landlocked', 'access_to_airport', 'access_to_port',
             'access_to_highway', 'access_to_railway', 'flood_risk_class',
             'seismic_hazard_zone', 'tropical_cyclone_wind_risk', 'tornadoes_wind_risk',
             'country_enc', 'region_enc', 'koppen_enc',
             't', 't_norm', 'sin_q', 'cos_q']
for c in num_feats:
    if c in train.columns:
        feature_cols.append(c)

# add raw categorical strings for CatBoost
# DO NOT include `location_id` as a feature — it causes memorization.
cat_str_feats = ['country_str', 'region_str', 'koppen_str']
for c in cat_str_feats:
    if c in train.columns:
        feature_cols.append(c)

X = train[feature_cols].copy()
y = train['construction_cost_per_m2_usd']

print(f'🎯 LEADERBOARD-ALIGNED CV (spatial GroupKFold by geolocation_name)')
print(f'Features: {len(feature_cols)} (NO location ID; country/region strings kept for CatBoost)')
print(f'Strategy: GroupKFold by geolocation_name across all years {train.t.min()}..{train.t.max()}')
print('='*60)
lgb_models = []
xgb_models = []
cat_models = []
ensemble_scores = []

# Spatial GroupKFold by geolocation_name to measure true spatial generalization
gkf = GroupKFold(n_splits=5)
for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y, groups=train['geolocation_name'])):
    X_tr = X.iloc[tr_idx].copy()
    X_val = X.iloc[val_idx].copy()
    y_tr = y.iloc[tr_idx]
    y_val = y.iloc[val_idx]

    print(f'\nFold {fold+1}: GroupKFold by geolocation_name -> Train: {len(X_tr)}, Val: {len(X_val)}')

    # LightGBM
    lgb_model = lgb.LGBMRegressor(
        n_estimators=3000,
        learning_rate=0.02,
        num_leaves=127,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.3,
        reg_lambda=0.3,
        min_child_samples=10,
        random_state=42 + fold,
        verbose=-1
    )
    lgb_model.fit(X_tr.select_dtypes(include=[np.number]), y_tr,
                  eval_set=[(X_val.select_dtypes(include=[np.number]), y_val)],
                  eval_metric='l1', callbacks=[lgb.early_stopping(150, verbose=False)])
    lgb_pred = lgb_model.predict(X_val.select_dtypes(include=[np.number]))
    lgb_models.append(lgb_model)

    # XGBoost
    xgb_model = xgb.XGBRegressor(
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.3,
        reg_lambda=0.3,
        min_child_weight=5,
        random_state=42 + fold
    )
    xgb_model.fit(X_tr.select_dtypes(include=[np.number]), y_tr)
    xgb_pred = xgb_model.predict(X_val.select_dtypes(include=[np.number]))
    xgb_models.append(xgb_model)

    # CatBoost uses raw string categorical columns; prepare feature matrix keeping object cols
    X_tr_cb = X_tr.copy()
    X_val_cb = X_val.copy()
    # ensure no NA in string cols
    for c in cat_str_feats:
        if c in X_tr_cb.columns:
            X_tr_cb[c] = X_tr_cb[c].fillna('missing')
            X_val_cb[c] = X_val_cb[c].fillna('missing')

    cat_features = [c for c in cat_str_feats if c in X_tr_cb.columns]
    # indices for CatBoost are positions within the full X_tr_cb columns
    cat_indices = [list(X_tr_cb.columns).index(c) for c in cat_features]

    cat_model = CatBoostRegressor(
        iterations=500,
        learning_rate=0.05,
        depth=7,
        l2_leaf_reg=3,
        random_seed=42 + fold,
        loss_function='MAE',
        verbose=0
    )
    cat_model.fit(X_tr_cb, y_tr, cat_features=cat_indices,
                  eval_set=(X_val_cb, y_val), early_stopping_rounds=50, verbose=False)
    cat_pred = cat_model.predict(X_val_cb)
    cat_models.append(cat_model)

    # Ensemble
    ensemble_pred = 0.4 * lgb_pred + 0.3 * xgb_pred + 0.3 * cat_pred
    mae = mean_absolute_error(y_val, ensemble_pred)
    ensemble_scores.append(mae)

    print(f'  Fold {fold+1} MAE: {mae:.2f}')

print(f'\n🎯 TIME-BASED CV MAE: {np.mean(ensemble_scores):.2f} ± {np.std(ensemble_scores):.2f}')

# Predictions on evaluation set
# Build X_eval following same feature order and types
X_eval = eval_df[feature_cols].copy()
for c in X_eval.select_dtypes(include=['object']).columns:
    X_eval[c] = X_eval[c].fillna('missing')
for c in X_eval.select_dtypes(include=[np.number]).columns:
    X_eval[c] = X_eval[c].fillna(0)

lgb_preds = np.mean([m.predict(X_eval.select_dtypes(include=[np.number])) for m in lgb_models], axis=0)
xgb_preds = np.mean([m.predict(X_eval.select_dtypes(include=[np.number])) for m in xgb_models], axis=0)
cat_preds = np.mean([m.predict(X_eval) for m in cat_models], axis=0)

final_preds = 0.4 * lgb_preds + 0.3 * xgb_preds + 0.3 * cat_preds

print(f'\n📊 Predictions:')
print(f'Mean: {final_preds.mean():.1f}, Std: {final_preds.std():.1f}')
print(f'Min: {final_preds.min():.1f}, Max: {final_preds.max():.1f}')

submission = pd.DataFrame({
    'data_id': eval_df['data_id'],
    'construction_cost_per_m2_usd': final_preds
})

submission.to_csv(BASE / 'submission_leaderboard_cv.csv', index=False)
print(f'\n✅ Saved: submission_leaderboard_cv.csv')
print(f'🎯 Expected LB: ~0.17-0.18 (matching your CV)')
