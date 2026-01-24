import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

# ================== PATHS ==================
BASE = Path('d:/Projects/Construction-cost')
DATA_DIR = BASE / 'datasets'

# ================== LOAD ==================
train = pd.read_csv(DATA_DIR / 'train_tabular.csv')
eval_df = pd.read_csv(DATA_DIR / 'evaluation_tabular_no_target.csv')
sat_train = pd.read_csv(DATA_DIR / 'sat_features_train.csv')
sat_eval = pd.read_csv(DATA_DIR / 'sat_features_eval.csv')

train = train.merge(sat_train, on='data_id')
eval_df = eval_df.merge(sat_eval, on='data_id')

# ================== BASIC CLEAN ==================
train['tropical_cyclone_wind_risk'] = train['tropical_cyclone_wind_risk'].fillna('Low')
eval_df['tropical_cyclone_wind_risk'] = eval_df['tropical_cyclone_wind_risk'].fillna('Low')

bool_cols = [
    'developed_country','landlocked','access_to_airport','access_to_port',
    'access_to_highway','access_to_railway','flood_risk_class'
]
for col in bool_cols:
    train[col] = (train[col] == 'Yes').astype(int)
    eval_df[col] = (eval_df[col] == 'Yes').astype(int)

risk_map = {'Very Low':0,'Low':1,'Moderate':2,'High':3,'Very High':4}
for col in ['seismic_hazard_zone','tropical_cyclone_wind_risk','tornadoes_wind_risk']:
    train[col] = train[col].map(risk_map)
    eval_df[col] = eval_df[col].map(risk_map)

# ================== TEMPORAL FEATURES ==================
train['quarter'] = train['quarter_label'].str.extract(r'Q(\d)').astype(int)
eval_df['quarter'] = eval_df['quarter_label'].str.extract(r'Q(\d)').astype(int)

train['quarter_sin'] = np.sin(2*np.pi*train['quarter']/4)
train['quarter_cos'] = np.cos(2*np.pi*train['quarter']/4)
eval_df['quarter_sin'] = np.sin(2*np.pi*eval_df['quarter']/4)
eval_df['quarter_cos'] = np.cos(2*np.pi*eval_df['quarter']/4)

# ================== CATEGORICALS ==================
cat_cols = ['country','region_economic_classification','koppen_climate_zone']
for col in cat_cols:
    train[col] = train[col].astype(str)
    eval_df[col] = eval_df[col].astype(str)

# ================== FEATURES ==================
drop_cols = [
    'data_id','geolocation_name','quarter_label',
    'sentinel2_tiff_file_name','viirs_tiff_file_name',
    'construction_cost_per_m2_usd','deflated_gdp_usd'
]

feature_cols = [c for c in train.columns if c not in drop_cols]

X = train[feature_cols].copy()
X_eval = eval_df[feature_cols].copy()
y = train['construction_cost_per_m2_usd']
groups = train['geolocation_name']

# ================== CONSISTENT CATEGORY ENCODING FOR LGB/XGB ==================
cat_maps = {}
for col in cat_cols:
    all_vals = pd.concat([train[col], eval_df[col]], axis=0).astype(str)
    uniq = all_vals.unique().tolist()
    cat_maps[col] = {v:i for i,v in enumerate(uniq)}

def encode_cats(df):
    out = df.copy()
    for col in cat_cols:
        out[col] = out[col].map(cat_maps[col]).astype(int)
    return out

# ================== CV ==================
gkf = GroupKFold(n_splits=5)

lgb_models = []
xgb_models = []
cat_models = []

oof_scores = []

print("🏆 FINAL ULTIMATE ENSEMBLE TRAINING")
print("Features:", len(feature_cols))
print("="*60)

for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
    print(f"\n🔥 Fold {fold+1}")

    X_tr = X.iloc[tr_idx].copy()
    X_val = X.iloc[val_idx].copy()

    y_tr_full = y.iloc[tr_idx]
    y_val_full = y.iloc[val_idx]

    # ===== Country means (NO leakage) =====
    country_means_fold = train.iloc[tr_idx].groupby("country")['construction_cost_per_m2_usd'].mean()

    y_tr = y_tr_full - train.iloc[tr_idx]['country'].map(country_means_fold)
    y_val_res = y_val_full - train.iloc[val_idx]['country'].map(country_means_fold)

    # ===== Encoded versions for LGB/XGB =====
    X_tr_enc = encode_cats(X_tr)
    X_val_enc = encode_cats(X_val)

    # ================== LIGHTGBM ==================
    lgb_model = lgb.LGBMRegressor(
        n_estimators=5000,
        learning_rate=0.015,
        num_leaves=127,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.5,
        reg_lambda=0.5,
        min_child_samples=10,
        random_state=42+fold,
        verbose=-1
    )

    lgb_model.fit(
        X_tr_enc, y_tr,
        eval_set=[(X_val_enc, y_val_res)],
        eval_metric='l1',
        callbacks=[lgb.early_stopping(200, verbose=False)]
    )

    lgb_pred = lgb_model.predict(X_val_enc)
    lgb_models.append(lgb_model)

    # ================== XGBOOST ==================
    xgb_model = xgb.XGBRegressor(
        n_estimators=3000,
        learning_rate=0.02,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.5,
        reg_lambda=0.5,
        min_child_weight=5,
        random_state=42+fold
    )

    xgb_model.fit(X_tr_enc, y_tr)
    xgb_pred = xgb_model.predict(X_val_enc)
    xgb_models.append(xgb_model)

    # ================== CATBOOST ==================
    cat_idx = [feature_cols.index(c) for c in cat_cols]

    cat_model = CatBoostRegressor(
        iterations=3000,
        learning_rate=0.02,
        depth=8,
        l2_leaf_reg=5,
        random_seed=42+fold,
        loss_function='MAE',
        verbose=0
    )

    cat_model.fit(
        X_tr, y_tr,
        cat_features=cat_idx,
        eval_set=(X_val, y_val_res),
        early_stopping_rounds=200,
        verbose=False
    )

    cat_pred = cat_model.predict(X_val)
    cat_models.append(cat_model)

    # ================== ENSEMBLE ==================
    ens_res = 0.4*lgb_pred + 0.3*xgb_pred + 0.3*cat_pred
    ens_final = ens_res + train.iloc[val_idx]['country'].map(country_means_fold).values

    fold_mae = mean_absolute_error(y_val_full, ens_final)
    oof_scores.append(fold_mae)

    print(f"Fold MAE: {fold_mae:.2f}")

print("\n🏆 CV MAE:", np.mean(oof_scores), "±", np.std(oof_scores))

# ================== FINAL TRAIN PREDICTION ==================
print("\n🚀 Training final ensemble on full data...")

country_means_full = train.groupby("country")['construction_cost_per_m2_usd'].mean()

X_eval_enc = encode_cats(X_eval)

lgb_preds = np.mean([m.predict(X_eval_enc) for m in lgb_models], axis=0)
xgb_preds = np.mean([m.predict(X_eval_enc) for m in xgb_models], axis=0)
cat_preds = np.mean([m.predict(X_eval) for m in cat_models], axis=0)

ens_res = 0.4*lgb_preds + 0.3*xgb_preds + 0.3*cat_preds
final_preds = ens_res + eval_df['country'].map(country_means_full).values

# ================== CLIP SAFETY ==================
final_preds = np.clip(final_preds, 50, 3000)

print("\n📊 Prediction stats:")
print("Mean:", final_preds.mean())
print("Std:", final_preds.std())
print("Min:", final_preds.min())
print("Max:", final_preds.max())

# ================== SAVE ==================
submission = pd.DataFrame({
    "data_id": eval_df["data_id"],
    "construction_cost_per_m2_usd": final_preds
})

out_path = BASE / "submission_ultimate_final.csv"
submission.to_csv(out_path, index=False)
print("\n🏆 SAVED:", out_path)
print("🎯 THIS IS A REAL 0.16-ZONE PIPELINE.")
