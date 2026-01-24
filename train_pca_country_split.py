"""
🚀 CORRECT PIPELINE: PCA-CNN + COUNTRY SPLIT + RESIDUAL TARGETS
This is the ONLY way to get CNN features working with tree models.

Strategy:
  1. PCA compress CNN: 2048 → 128 dims
  2. Use ONLY tabular (~30) + PCA-CNN (~128) = ~160 features
  3. Split by country + residual targets
  4. Train per-country models
  5. Single LightGBM (no ensemble yet)

Expected result: CV MAE ~90-110 ✅
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import lightgbm as lgb
from tqdm import tqdm

BASE = Path("d:/Projects/Construction-cost")
DATA_DIR = BASE / "datasets"
CNN_DIR = BASE / "cache" / "cnn_feats"

print("="*70)
print("🎯 PIPELINE: PCA-CNN + COUNTRY SPLIT + RESIDUAL TARGETS")
print("="*70)

# Load data
train = pd.read_csv(DATA_DIR / "train_tabular.csv")
eval_df = pd.read_csv(DATA_DIR / "evaluation_tabular_no_target.csv")

# Build tabular features
def build_tabular(df):
    df = df.copy()
    df['log_gdp'] = np.log1p(df['deflated_gdp_usd'])
    df['log_dist'] = np.log1p(df['straight_distance_to_capital_km'])

    bool_cols = ['developed_country','landlocked','access_to_airport','access_to_port',
                 'access_to_highway','access_to_railway']
    for c in bool_cols:
        df[c] = (df[c] == "Yes").astype(int)

    df['infra_score'] = df[bool_cols].sum(axis=1)

    return df

train = build_tabular(train)
eval_df = build_tabular(eval_df)

# 📥 Load CNN features
def load_cnn_feat(fname):
    key = Path(fname).stem
    path = CNN_DIR / f"{key}.npy"
    if path.exists():
        return np.load(path)
    else:
        return np.zeros(2048)

print("\n📥 Loading CNN features (2048-dim)...")
train_cnn = np.vstack([load_cnn_feat(f) for f in tqdm(train['sentinel2_tiff_file_name'], desc="Train")])
eval_cnn  = np.vstack([load_cnn_feat(f) for f in tqdm(eval_df['sentinel2_tiff_file_name'], desc="Eval")])

print(f"   Train CNN shape: {train_cnn.shape}")
print(f"   Eval CNN shape:  {eval_cnn.shape}")

# 🔧 PCA COMPRESSION: 2048 → 128
print("\n🔧 Compressing CNN features with PCA...")
scaler = StandardScaler()
train_cnn_scaled = scaler.fit_transform(train_cnn)
eval_cnn_scaled = scaler.transform(eval_cnn)

pca = PCA(n_components=128, random_state=42)
train_cnn_pca = pca.fit_transform(train_cnn_scaled)
eval_cnn_pca = pca.transform(eval_cnn_scaled)

explained_var = pca.explained_variance_ratio_.sum()
print(f"   ✅ CNN dims: {train_cnn.shape[1]} → {train_cnn_pca.shape[1]}")
print(f"   📊 Explained variance: {explained_var:.4f} ({explained_var*100:.2f}%)")

# Extract tabular features (NO SAT FEATURES)
drop_cols = ['construction_cost_per_m2_usd','sentinel2_tiff_file_name','viirs_tiff_file_name',
             'geolocation_name','quarter_label','data_id','country','deflated_gdp_usd']

tab_feats = [c for c in train.columns if c not in drop_cols and train[c].dtype != 'object']
X_tab = train[tab_feats].fillna(0).values
X_eval_tab = eval_df[tab_feats].fillna(0).values

print(f"\n📊 Feature dimensions:")
print(f"   Tabular features: {X_tab.shape[1]}")
print(f"   PCA-CNN features: {train_cnn_pca.shape[1]}")
print(f"   Total features:   {X_tab.shape[1] + train_cnn_pca.shape[1]}")

# Combine tabular + PCA-CNN
X = np.hstack([X_tab, train_cnn_pca])
X_eval = np.hstack([X_eval_tab, eval_cnn_pca])

y = train['construction_cost_per_m2_usd'].values
groups = train['geolocation_name'].values
countries = train['country'].values
country_eval = eval_df['country'].values

print(f"   Train shape: {X.shape}, Eval shape: {X_eval.shape}")

# 🌍 COUNTRY-WISE ANALYSIS
print(f"\n🌍 Country analysis:")
unique_countries = np.unique(countries)
for country in sorted(unique_countries):
    mask = countries == country
    n_samples = mask.sum()
    mean_cost = y[mask].mean()
    std_cost = y[mask].std()
    print(f"   {country:15s}: {n_samples:4d} samples, mean=${mean_cost:8.2f}, std=${std_cost:7.2f}")

# 🔴 TRAIN SEPARATE MODELS PER COUNTRY WITH RESIDUAL TARGETS
print(f"\n{'='*70}")
print("🔴 TRAINING PER-COUNTRY MODELS WITH RESIDUAL TARGETS")
print(f"{'='*70}\n")

# First, compute country means on full training data
country_means = {}
for country in sorted(unique_countries):
    mask = countries == country
    country_means[country] = y[mask].mean()
    print(f"   Country mean {country}: ${country_means[country]:.2f}")

# Train per-country models
country_models = {}
country_cv_maes = {}

for country in sorted(unique_countries):
    print(f"\n🔷 Training model for {country}...")
    
    # Get country subset
    mask = countries == country
    X_c = X[mask]
    y_c = y[mask]
    groups_c = groups[mask]
    
    # Residual target
    country_mean = country_means[country]
    y_resid = y_c - country_mean
    
    print(f"   Samples: {X_c.shape[0]}, Features: {X_c.shape[1]}")
    print(f"   Target mean (residuals): {y_resid.mean():.2f}")
    
    # Cross-validation per country
    gkf = GroupKFold(n_splits=min(5, len(np.unique(groups_c))))
    fold_maes = []
    oof_preds = np.zeros(len(X_c))
    models_list = []
    
    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_c, y_resid, groups_c)):
        X_tr, X_val = X_c[tr_idx], X_c[val_idx]
        y_tr, y_val = y_resid[tr_idx], y_resid[val_idx]
        
        model = lgb.LGBMRegressor(
            n_estimators=3000,
            learning_rate=0.03,
            num_leaves=256,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42 + fold,
            verbose=-1,
            n_jobs=-1
        )
        
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric='l1',
            callbacks=[lgb.early_stopping(150, verbose=False)]
        )
        
        pred_resid = model.predict(X_val)
        oof_preds[val_idx] = pred_resid
        
        # Compute MAE on FINAL predictions (residual + mean)
        pred_final = pred_resid + country_mean
        true_final = y_val + country_mean
        mae = mean_absolute_error(true_final, pred_final)
        fold_maes.append(mae)
        models_list.append(model)
        
        print(f"   Fold {fold+1} MAE: {mae:.2f}")
    
    # Overall country CV MAE
    oof_final = oof_preds + country_mean
    true_final_all = y_c
    country_cv_mae = mean_absolute_error(true_final_all, oof_final)
    country_cv_maes[country] = country_cv_mae
    
    print(f"   ✅ Country CV MAE: {country_cv_mae:.2f}")
    
    country_models[country] = {
        'models': models_list,
        'mean': country_mean,
        'cv_mae': country_cv_mae
    }

# 🎯 OVERALL CV RESULTS
print(f"\n{'='*70}")
print("📊 CROSS-VALIDATION RESULTS")
print(f"{'='*70}")

weighted_cv_mae = sum(
    country_cv_maes[country] * (countries == country).sum() 
    for country in sorted(unique_countries)
) / len(countries)

print(f"\nPer-country CV MAE:")
for country in sorted(unique_countries):
    print(f"   {country:15s}: {country_cv_maes[country]:8.2f}")

print(f"\n🎯 WEIGHTED CV MAE: {weighted_cv_mae:.2f}")
print(f"📊 Explained variance (CNN PCA): {explained_var*100:.2f}%")

# 🎲 PREDICTIONS
print(f"\n{'='*70}")
print("🎲 GENERATING PREDICTIONS")
print(f"{'='*70}\n")

eval_preds = np.zeros(len(X_eval))

for country in sorted(unique_countries):
    mask_eval = country_eval == country
    n_eval = mask_eval.sum()
    
    if n_eval == 0:
        print(f"   {country}: No eval samples")
        continue
    
    X_eval_c = X_eval[mask_eval]
    country_mean = country_models[country]['mean']
    
    # Average predictions from all folds
    pred_resid = np.mean(
        [m.predict(X_eval_c) for m in country_models[country]['models']],
        axis=0
    )
    pred_final = pred_resid + country_mean
    eval_preds[mask_eval] = pred_final
    
    print(f"   {country:15s}: {n_eval:4d} samples, pred range [${pred_final.min():.1f}, ${pred_final.max():.1f}]")

print(f"\n📈 Final predictions:")
print(f"   Mean: ${eval_preds.mean():.2f}")
print(f"   Std:  ${eval_preds.std():.2f}")
print(f"   Min:  ${eval_preds.min():.2f}")
print(f"   Max:  ${eval_preds.max():.2f}")

# Save submission
sub = pd.DataFrame({
    "data_id": eval_df["data_id"],
    "construction_cost_per_m2_usd": eval_preds
})

sub.to_csv(BASE / "submission_pca_country_split.csv", index=False)
print(f"\n✅ Submission saved: submission_pca_country_split.csv")

# Summary
print(f"\n{'='*70}")
print("📌 SUMMARY")
print(f"{'='*70}")
print(f"✅ PCA compression: 2048 → 128 dims ({explained_var*100:.2f}% variance)")
print(f"✅ Features: {X.shape[1]} (tabular + PCA-CNN)")
print(f"✅ Per-country residual modeling: {len(unique_countries)} countries")
print(f"🎯 Weighted CV MAE: {weighted_cv_mae:.2f}")
print(f"{'='*70}\n")
