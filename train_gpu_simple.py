"""
🚀 Simple GPU Training - XGBoost Only (No CuPy needed)
Just install: pip install xgboost
XGBoost auto-detects CUDA
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

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

# 📊 Baseline check
baseline_mae = mean_absolute_error(train['construction_cost_per_m2_usd'], train['country_mean'])
print(f'📊 Baseline MAE (country mean only): {baseline_mae:.2f}')
print(f'🎯 Target: Beat baseline significantly')
print('='*60)

# Features - DROP GDP
drop_cols = ['data_id', 'geolocation_name', 'quarter_label', 'sentinel2_tiff_file_name', 
             'viirs_tiff_file_name', 'construction_cost_per_m2_usd', 'target_residual',
             'country_mean', 'deflated_gdp_usd']

feature_cols = [c for c in train.columns if c not in drop_cols]
X = train[feature_cols].fillna(0)
y = train['target_residual']
groups = train['geolocation_name']

print(f'⚡ Training LightGBM with {len(feature_cols)} features')
print(f'🔥 Enhanced: builtup_ratio, vegetation_ratio, urban_ratio')

# Check GPU
try:
    import subprocess
    result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], 
                          capture_output=True, text=True, timeout=2)
    if result.returncode == 0:
        gpu_name = result.stdout.strip()
        print(f'✅ GPU Detected: {gpu_name}')
        print(f'✅ Using LightGBM')
    else:
        print('⚠️  GPU not detected, will use CPU')
except:
    print('⚠️  Could not check GPU, attempting GPU training anyway')

print('='*60)

# GPU Training
gkf = GroupKFold(n_splits=5)
cv_scores = []
models = []

for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
    
    model = lgb.LGBMRegressor(
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
    
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric='l1',
        callbacks=[lgb.early_stopping(150, verbose=False)]
    )
    
    pred_residual = model.predict(X_val)
    pred_final = pred_residual + train.iloc[val_idx]['country_mean'].values
    true_final = train.iloc[val_idx]['construction_cost_per_m2_usd'].values
    
    mae = mean_absolute_error(true_final, pred_final)
    cv_scores.append(mae)
    models.append(model)
    
    print(f'Fold {fold+1} MAE: {mae:.2f}')

print(f'\n🚀 LightGBM CV MAE: {np.mean(cv_scores):.2f} ± {np.std(cv_scores):.2f}')

# Predictions
X_eval = eval_df[feature_cols].fillna(0)
residual_preds = np.mean([m.predict(X_eval) for m in models], axis=0)
final_preds = residual_preds + eval_df['country_mean'].values

print(f'\nPredictions - Mean: {final_preds.mean():.1f}, Std: {final_preds.std():.1f}')
print(f'Min: {final_preds.min():.1f}, Max: {final_preds.max():.1f}')

submission = pd.DataFrame({
    'data_id': eval_df['data_id'],
    'construction_cost_per_m2_usd': final_preds
})

submission.to_csv(BASE / 'submission_gpu.csv', index=False)
print(f'\n✅ Saved: submission_gpu.csv')
