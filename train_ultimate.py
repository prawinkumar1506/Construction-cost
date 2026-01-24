"""
🏆 ULTIMATE ENSEMBLE: LightGBM + XGBoost + CatBoost
Target: 0.16 zone
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

BASE = Path('d:/Projects/Construction-cost')
DATA_DIR = BASE / 'datasets'

# Load data
train = pd.read_csv(DATA_DIR / 'train_tabular.csv')
eval_df = pd.read_csv(DATA_DIR / 'evaluation_tabular_no_target.csv')
sat_train = pd.read_csv(DATA_DIR / 'sat_features_train.csv')
sat_eval = pd.read_csv(DATA_DIR / 'sat_features_eval.csv')

train = train.merge(sat_train, on='data_id')
eval_df = eval_df.merge(sat_eval, on='data_id')

# Preprocessing
train['tropical_cyclone_wind_risk'] = train['tropical_cyclone_wind_risk'].fillna('Low')
eval_df['tropical_cyclone_wind_risk'] = eval_df['tropical_cyclone_wind_risk'].fillna('Low')

bool_cols = ['developed_country', 'landlocked', 'access_to_airport', 'access_to_port', 
             'access_to_highway', 'access_to_railway', 'flood_risk_class']
for col in bool_cols:
    train[col] = (train[col] == 'Yes').astype(int)
    eval_df[col] = (eval_df[col] == 'Yes').astype(int)

risk_map = {'Very Low': 0, 'Low': 1, 'Moderate': 2, 'High': 3, 'Very High': 4}
for col in ['seismic_hazard_zone', 'tropical_cyclone_wind_risk', 'tornadoes_wind_risk']:
    train[col] = train[col].map(risk_map)
    eval_df[col] = eval_df[col].map(risk_map)

for col in ['country', 'region_economic_classification', 'koppen_climate_zone']:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    eval_df[col] = le.transform(eval_df[col])

# 🔥 Country residual modeling + LOG TRANSFORM
country_means = train.groupby('country')['construction_cost_per_m2_usd'].mean()
train['country_mean'] = train['country'].map(country_means)
train['target_residual'] = train['construction_cost_per_m2_usd'] - train['country_mean']
train['target_log'] = np.log1p(train['target_residual'] + 1000)  # Shift to positive
eval_df['country_mean'] = eval_df['country'].map(country_means)

baseline_mae = mean_absolute_error(train['construction_cost_per_m2_usd'], train['country_mean'])
print(f'📊 Baseline MAE: {baseline_mae:.2f}')

# Features
drop_cols = ['data_id', 'geolocation_name', 'quarter_label', 'sentinel2_tiff_file_name', 
             'viirs_tiff_file_name', 'construction_cost_per_m2_usd', 'target_residual',
             'country_mean', 'deflated_gdp_usd', 'target_log']

feature_cols = [c for c in train.columns if c not in drop_cols]
X = train[feature_cols].fillna(0)
y_log = train['target_log']
groups = train['geolocation_name']

print(f'🏆 ULTIMATE ENSEMBLE')
print(f'Features: {len(feature_cols)}')
print(f'🔥 Log transform + Country residual')
print('='*60)

gkf = GroupKFold(n_splits=5)
lgb_models = []
xgb_models = []
cat_models = []
ensemble_scores = []

for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y_log, groups)):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y_log.iloc[tr_idx], y_log.iloc[val_idx]
    
    # LightGBM
    lgb_model = lgb.LGBMRegressor(
        n_estimators=5000,
        learning_rate=0.015,
        num_leaves=127,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.5,
        reg_lambda=0.5,
        min_child_samples=10,
        random_state=42 + fold,
        verbose=-1
    )
    lgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric='l1',
                  callbacks=[lgb.early_stopping(200, verbose=False)])
    lgb_pred_log = lgb_model.predict(X_val)
    lgb_models.append(lgb_model)
    
    # XGBoost
    xgb_model = xgb.XGBRegressor(
        n_estimators=3000,
        learning_rate=0.02,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.5,
        reg_lambda=0.5,
        min_child_weight=5,
        random_state=42 + fold
    )
    xgb_model.fit(X_tr, y_tr)
    xgb_pred_log = xgb_model.predict(X_val)
    xgb_models.append(xgb_model)
    
    # CatBoost
    cat_model = CatBoostRegressor(
        iterations=3000,
        learning_rate=0.02,
        depth=8,
        l2_leaf_reg=5,
        random_seed=42 + fold,
        verbose=0
    )
    cat_model.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=200, verbose=False)
    cat_pred_log = cat_model.predict(X_val)
    cat_models.append(cat_model)
    
    # Ensemble (weighted average)
    ensemble_pred_log = 0.4 * lgb_pred_log + 0.3 * xgb_pred_log + 0.3 * cat_pred_log
    
    # Convert back to original scale
    ensemble_pred_residual = np.expm1(ensemble_pred_log) - 1000
    ensemble_pred_final = ensemble_pred_residual + train.iloc[val_idx]['country_mean'].values
    true_final = train.iloc[val_idx]['construction_cost_per_m2_usd'].values
    
    mae = mean_absolute_error(true_final, ensemble_pred_final)
    ensemble_scores.append(mae)
    
    print(f'Fold {fold+1} MAE: {mae:.2f}')

print(f'\n🏆 ENSEMBLE CV MAE: {np.mean(ensemble_scores):.2f} ± {np.std(ensemble_scores):.2f}')
print(f'📊 Improvement over baseline: {baseline_mae - np.mean(ensemble_scores):.2f}')

# Predictions
X_eval = eval_df[feature_cols].fillna(0)

lgb_preds_log = np.mean([m.predict(X_eval) for m in lgb_models], axis=0)
xgb_preds_log = np.mean([m.predict(X_eval) for m in xgb_models], axis=0)
cat_preds_log = np.mean([m.predict(X_eval) for m in cat_models], axis=0)

# Weighted ensemble
ensemble_preds_log = 0.4 * lgb_preds_log + 0.3 * xgb_preds_log + 0.3 * cat_preds_log

# Convert back
ensemble_preds_residual = np.expm1(ensemble_preds_log) - 1000
final_preds = ensemble_preds_residual + eval_df['country_mean'].values

print(f'\n📊 Predictions:')
print(f'Mean: {final_preds.mean():.1f}, Std: {final_preds.std():.1f}')
print(f'Min: {final_preds.min():.1f}, Max: {final_preds.max():.1f}')

submission = pd.DataFrame({
    'data_id': eval_df['data_id'],
    'construction_cost_per_m2_usd': final_preds
})

submission.to_csv(BASE / 'submission_ultimate.csv', index=False)
print(f'\n🏆 Saved: submission_ultimate.csv')
print(f'🎯 Target: 0.16 zone!')
