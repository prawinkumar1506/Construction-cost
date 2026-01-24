"""
🏆 PRODUCTION-GRADE FINETUNED PIPELINE
Target: 0.16 leaderboard score
Strategy: Ensemble + Advanced hyperparameter tuning + Feature engineering
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler, RobustScaler
import lightgbm as lgb
import catboost as cb
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

BASE = Path("d:/Projects/Construction-cost")
DATA_DIR = BASE / "datasets"
CNN_DIR = BASE / "cache" / "cnn_feats"

print("\n" + "="*70)
print("🏆 PRODUCTION FINETUNED PIPELINE - TARGET: 0.16")
print("="*70 + "\n")

# Load data
train = pd.read_csv(DATA_DIR / "train_tabular.csv")
eval_df = pd.read_csv(DATA_DIR / "evaluation_tabular_no_target.csv")

def build_advanced_features(df):
    """Advanced feature engineering"""
    df = df.copy()
    
    # Log transformations
    df['log_gdp'] = np.log1p(df['deflated_gdp_usd'])
    df['log_dist'] = np.log1p(df['straight_distance_to_capital_km'])
    
    # Boolean encoding
    bool_cols = ['developed_country','landlocked','access_to_airport','access_to_port',
                 'access_to_highway','access_to_railway']
    for c in bool_cols:
        df[c] = (df[c] == "Yes").astype(int)
    
    # Infrastructure score
    df['infra_score'] = df[bool_cols].sum(axis=1)
    
    # Climate interactions
    climate_dummies = pd.get_dummies(df['koppen_climate_zone'], prefix='climate')
    df = pd.concat([df, climate_dummies], axis=1)
    
    # Risk interactions
    df['natural_hazard_risk'] = 0
    if 'flood_risk_class' in df.columns:
        df['natural_hazard_risk'] += (df['flood_risk_class'] == "Yes").astype(int)
    df['natural_hazard_risk'] += (df['tropical_cyclone_wind_risk'] == "High").astype(int)
    
    return df

train = build_advanced_features(train)
eval_df = build_advanced_features(eval_df)

# Load CNN features
def load_cnn_feat(fname):
    key = Path(fname).stem
    path = CNN_DIR / f"{key}.npy"
    if path.exists():
        return np.load(path)
    else:
        return np.zeros(2048)

print("📥 Loading feature sets...")
train_cnn = np.vstack([load_cnn_feat(f) for f in tqdm(train['sentinel2_tiff_file_name'], desc="Train CNN")])
eval_cnn = np.vstack([load_cnn_feat(f) for f in tqdm(eval_df['sentinel2_tiff_file_name'], desc="Eval CNN")])

# Satellite spectral features
sat_train = pd.read_csv(DATA_DIR / "sat_features_train.csv")
sat_eval = pd.read_csv(DATA_DIR / "sat_features_eval.csv")

sat_cols = [c for c in sat_train.columns if c != 'data_id']
sat_train_feat = sat_train[sat_cols].fillna(0).values
sat_eval_feat = sat_eval[sat_cols].fillna(0).values

# Tabular features
drop_cols = ['construction_cost_per_m2_usd','sentinel2_tiff_file_name','viirs_tiff_file_name',
             'geolocation_name','quarter_label','data_id','country','koppen_climate_zone']

tab_feats = [c for c in train.columns if c not in drop_cols and 
             (train[c].dtype in [np.float64, np.int64] or c.startswith('climate'))]

X_tab = train[tab_feats].fillna(0).values
X_eval_tab = eval_df[tab_feats].fillna(0).values

# Feature scaling
print("📊 Feature scaling...")
scaler_tab = StandardScaler()
X_tab = scaler_tab.fit_transform(X_tab)
X_eval_tab = scaler_tab.transform(X_eval_tab)

scaler_sat = RobustScaler()
sat_train_feat = scaler_sat.fit_transform(sat_train_feat)
sat_eval_feat = scaler_sat.transform(sat_eval_feat)

scaler_cnn = RobustScaler()
train_cnn = scaler_cnn.fit_transform(train_cnn)
eval_cnn = scaler_cnn.transform(eval_cnn)

# Combine all features
X = np.hstack([X_tab, sat_train_feat, train_cnn])
X_eval = np.hstack([X_eval_tab, sat_eval_feat, eval_cnn])

y = train['construction_cost_per_m2_usd'].values
groups = train['geolocation_name'].values

print(f"\n{'='*70}")
print(f"📊 ADVANCED FEATURE SUMMARY")
print(f"{'='*70}")
print(f"  Tabular + engineered:  {X_tab.shape[1]:4d}")
print(f"  Satellite spectral:    {sat_train_feat.shape[1]:4d}")
print(f"  CNN embeddings:        {train_cnn.shape[1]:4d}")
print(f"  TOTAL FEATURES:        {X.shape[1]:4d} ⭐")
print(f"  Training samples:      {X.shape[0]:4d}")
print(f"  Eval samples:          {X_eval.shape[0]:4d}")
print(f"{'='*70}\n")

# Advanced CV with ensembling
gkf = GroupKFold(n_splits=5)
oof_lgb = np.zeros(len(X))
oof_cat = np.zeros(len(X))
models_lgb = []
models_cat = []
fold_results = []

for fold, (tr, val) in enumerate(gkf.split(X, y, groups)):
    print(f"\n🔄 Fold {fold+1}/5")
    print("  Training LightGBM...")
    
    # FINETUNED LightGBM
    model_lgb = lgb.LGBMRegressor(
        n_estimators=3000,
        learning_rate=0.02,
        num_leaves=384,
        max_depth=8,
        subsample=0.75,
        subsample_freq=5,
        colsample_bytree=0.75,
        reg_alpha=0.5,
        reg_lambda=2.0,
        min_child_samples=5,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    model_lgb.fit(
        X[tr], y[tr],
        eval_set=[(X[val], y[val])],
        eval_metric="l1",
        callbacks=[lgb.early_stopping(300, verbose=False)]
    )
    
    pred_lgb = model_lgb.predict(X[val])
    oof_lgb[val] = pred_lgb
    mae_lgb = mean_absolute_error(y[val], pred_lgb)
    
    print(f"  Training CatBoost...")
    
    # FINETUNED CatBoost
    model_cat = cb.CatBoostRegressor(
        iterations=2000,
        learning_rate=0.05,
        depth=7,
        l2_leaf_reg=5.0,
        subsample=0.8,
        colsample_bylevel=0.8,
        random_state=42,
        verbose=False,
        loss_function='MAE'
    )
    
    model_cat.fit(X[tr], y[tr], eval_set=(X[val], y[val]), early_stopping_rounds=200)
    
    pred_cat = model_cat.predict(X[val])
    oof_cat[val] = pred_cat
    mae_cat = mean_absolute_error(y[val], pred_cat)
    
    # Ensemble prediction
    pred_ens = (pred_lgb + pred_cat) / 2
    mae_ens = mean_absolute_error(y[val], pred_ens)
    
    fold_results.append({
        'fold': fold + 1,
        'lgb_mae': mae_lgb,
        'cat_mae': mae_cat,
        'ens_mae': mae_ens
    })
    
    print(f"  ✓ LightGBM MAE:  {mae_lgb:.6f}")
    print(f"  ✓ CatBoost MAE:  {mae_cat:.6f}")
    print(f"  ✓ Ensemble MAE:  {mae_ens:.6f} ⭐")
    
    models_lgb.append(model_lgb)
    models_cat.append(model_cat)

# Final ensemble OOF
oof_ens = (oof_lgb + oof_cat) / 2
cv_mae_lgb = mean_absolute_error(y, oof_lgb)
cv_mae_cat = mean_absolute_error(y, oof_cat)
cv_mae_ens = mean_absolute_error(y, oof_ens)

print(f"\n{'='*70}")
print(f"🎯 CROSS-VALIDATION FINAL RESULTS")
print(f"{'='*70}")
for res in fold_results:
    print(f"  Fold {res['fold']}: LGB {res['lgb_mae']:.6f} | CAT {res['cat_mae']:.6f} | ENS {res['ens_mae']:.6f}")
print(f"{'='*70}")
print(f"  LightGBM CV MAE:  {cv_mae_lgb:.6f}")
print(f"  CatBoost CV MAE:  {cv_mae_cat:.6f}")
print(f"  Ensemble CV MAE:  {cv_mae_ens:.6f} ⭐ TARGET: 0.16")
print(f"{'='*70}\n")

# Final predictions - weighted ensemble
print("🎲 Generating final ensemble predictions...")
pred_eval_lgb = np.mean([m.predict(X_eval) for m in models_lgb], axis=0)
pred_eval_cat = np.mean([m.predict(X_eval) for m in models_cat], axis=0)
pred_eval = (pred_eval_lgb * 0.5 + pred_eval_cat * 0.5)  # 50/50 blend

# Save multiple submissions for testing
submissions = {
    'submission_cnn_hybrid_lgb.csv': pred_eval_lgb,
    'submission_cnn_hybrid_cat.csv': pred_eval_cat,
    'submission_cnn_hybrid_ens.csv': pred_eval,
}

for fname, preds in submissions.items():
    sub = pd.DataFrame({
        "data_id": eval_df["data_id"],
        "construction_cost_per_m2_usd": preds
    })
    sub.to_csv(BASE / fname, index=False)
    print(f"✅ {fname:40s} | Range: [{preds.min():8.2f}, {preds.max():8.2f}]")

print(f"\n{'='*70}")
print(f"✨ PRODUCTION PIPELINE COMPLETE")
print(f"{'='*70}")
print(f"📊 Best CV Score: {cv_mae_ens:.6f}")
print(f"🎯 Expected LB Score: ~0.17-0.19 (on this dataset)")
print(f"💡 Next steps for 0.16:")
print(f"   1. Complete CNN feature extraction for ALL images")
print(f"   2. Use larger CNN models (EfficientNet, ConvNext)")
print(f"   3. Hyperparameter grid search")
print(f"   4. Add VIIRS features")
print(f"   5. Create stacked meta-models")
print(f"{'='*70}\n")
