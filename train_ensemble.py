"""
🚀 Ensemble: LightGBM + XGBoost
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb

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

# Country residual modeling
country_means = train.groupby('country')['construction_cost_per_m2_usd'].mean()
train['country_mean'] = train['country'].map(country_means)
train['target_residual'] = train['construction_cost_per_m2_usd'] - train['country_mean']
eval_df['country_mean'] = eval_df['country'].map(country_means)

# Features
drop_cols = ['data_id', 'geolocation_name', 'quarter_label', 'sentinel2_tiff_file_name', 
             'viirs_tiff_file_name', 'construction_cost_per_m2_usd', 'target_residual',
             'country_mean', 'deflated_gdp_usd']

feature_cols = [c for c in train.columns if c not in drop_cols]
X = train[feature_cols].fillna(0)
y = train['target_residual']
groups = train['geolocation_name']

print(f'🚀 Ensemble Training: LightGBM + XGBoost')
print(f'Features: {len(feature_cols)}')
print('='*60)

gkf = GroupKFold(n_splits=5)
lgb_models = []
xgb_models = []
lgb_scores = []
xgb_scores = []

for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
    
    # LightGBM
    lgb_model = lgb.LGBMRegressor(
        n_estimators=5000,
        learning_rate=0.02,
        num_leaves=127,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42 + fold,
        verbose=-1
    )
    lgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric='l1',
                  callbacks=[lgb.early_stopping(150, verbose=False)])
    
    lgb_pred = lgb_model.predict(X_val)
    lgb_final = lgb_pred + train.iloc[val_idx]['country_mean'].values
    lgb_mae = mean_absolute_error(train.iloc[val_idx]['construction_cost_per_m2_usd'].values, lgb_final)
    lgb_scores.append(lgb_mae)
    lgb_models.append(lgb_model)
    
    # XGBoost
    xgb_model = xgb.XGBRegressor(
        n_estimators=3000,
        learning_rate=0.03,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42 + fold
    )
    xgb_model.fit(X_tr, y_tr)
    
    xgb_pred = xgb_model.predict(X_val)
    xgb_final = xgb_pred + train.iloc[val_idx]['country_mean'].values
    xgb_mae = mean_absolute_error(train.iloc[val_idx]['construction_cost_per_m2_usd'].values, xgb_final)
    xgb_scores.append(xgb_mae)
    xgb_models.append(xgb_model)
    
    # Ensemble
    ensemble_pred = (lgb_pred + xgb_pred) / 2
    ensemble_final = ensemble_pred + train.iloc[val_idx]['country_mean'].values
    ensemble_mae = mean_absolute_error(train.iloc[val_idx]['construction_cost_per_m2_usd'].values, ensemble_final)
    
    print(f'Fold {fold+1} - LGB: {lgb_mae:.2f}, XGB: {xgb_mae:.2f}, Ensemble: {ensemble_mae:.2f}')

print(f'\n🚀 LightGBM CV: {np.mean(lgb_scores):.2f} ± {np.std(lgb_scores):.2f}')
print(f'🚀 XGBoost CV: {np.mean(xgb_scores):.2f} ± {np.std(xgb_scores):.2f}')

# Predictions
X_eval = eval_df[feature_cols].fillna(0)

lgb_preds = np.mean([m.predict(X_eval) for m in lgb_models], axis=0)
xgb_preds = np.mean([m.predict(X_eval) for m in xgb_models], axis=0)

# Ensemble (average)
ensemble_preds = (lgb_preds + xgb_preds) / 2
final_preds = ensemble_preds + eval_df['country_mean'].values

print(f'\nPredictions - Mean: {final_preds.mean():.1f}, Std: {final_preds.std():.1f}')
print(f'Min: {final_preds.min():.1f}, Max: {final_preds.max():.1f}')

submission = pd.DataFrame({
    'data_id': eval_df['data_id'],
    'construction_cost_per_m2_usd': final_preds
})

submission.to_csv(BASE / 'submission_ensemble.csv', index=False)
print(f'\n✅ Saved: submission_ensemble.csv')
