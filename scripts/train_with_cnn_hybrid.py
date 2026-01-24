import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
from tqdm import tqdm

BASE = Path("d:/Projects/Construction-cost")
DATA_DIR = BASE / "datasets"
CNN_DIR = BASE / "cache" / "cnn_feats"

print("="*60)
print("🧠 TRAINING WITH CNN + TABULAR FEATURES")
print("="*60)

# Load data
train = pd.read_csv(DATA_DIR / "train_tabular.csv")
eval_df = pd.read_csv(DATA_DIR / "evaluation_tabular_no_target.csv")

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

# Load CNN features (or use zeros if not cached)
def load_cnn_feat(fname):
    key = Path(fname).stem
    path = CNN_DIR / f"{key}.npy"
    if path.exists():
        return np.load(path)
    else:
        return np.zeros(2048)

print("\n📥 Loading CNN features...")
train_cnn = np.vstack([load_cnn_feat(f) for f in tqdm(train['sentinel2_tiff_file_name'])])
eval_cnn  = np.vstack([load_cnn_feat(f) for f in tqdm(eval_df['sentinel2_tiff_file_name'])])

# Also load pre-extracted satellite features
print("📥 Loading pre-extracted satellite features...")
sat_train = pd.read_csv(DATA_DIR / "sat_features_train.csv")
sat_eval = pd.read_csv(DATA_DIR / "sat_features_eval.csv")

sat_cols = [c for c in sat_train.columns if c != 'data_id']
sat_train_feat = sat_train[sat_cols].fillna(0).values
sat_eval_feat = sat_eval[sat_cols].fillna(0).values

# Tabular features
drop_cols = ['construction_cost_per_m2_usd','sentinel2_tiff_file_name','viirs_tiff_file_name',
             'geolocation_name','quarter_label','data_id','country']

tab_feats = [c for c in train.columns if c not in drop_cols and train[c].dtype != 'object']

X_tab = train[tab_feats].fillna(0).values
X_eval_tab = eval_df[tab_feats].fillna(0).values

# Final features: tabular + satellite spectral + CNN embeddings
X = np.hstack([X_tab, sat_train_feat, train_cnn])
X_eval = np.hstack([X_eval_tab, sat_eval_feat, eval_cnn])

y = train['construction_cost_per_m2_usd'].values
groups = train['geolocation_name'].values

print(f"\n{'='*60}")
print(f"📊 Feature Engineering Summary")
print(f"{'='*60}")
print(f"  Tabular features:         {X_tab.shape[1]:4d}")
print(f"  Satellite spectral feat:  {sat_train_feat.shape[1]:4d}")
print(f"  CNN embeddings:           {train_cnn.shape[1]:4d}")
print(f"  Total features:           {X.shape[1]:4d}")
print(f"  Training samples:         {X.shape[0]:4d}")
print(f"  Eval samples:             {X_eval.shape[0]:4d}")
print(f"{'='*60}\n")

# Cross-validation
gkf = GroupKFold(n_splits=5)
oof = np.zeros(len(X))
models = []
fold_maes = []

for fold, (tr, val) in enumerate(gkf.split(X, y, groups)):
    print(f"🔄 Fold {fold+1}/5")

    model = lgb.LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=256,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )

    model.fit(
        X[tr], y[tr],
        eval_set=[(X[val], y[val])],
        eval_metric="l1",
        callbacks=[lgb.early_stopping(200, verbose=False)]
    )

    pred = model.predict(X[val])
    oof[val] = pred
    mae = mean_absolute_error(y[val], pred)
    fold_maes.append(mae)
    print(f"   ✓ Fold MAE: {mae:.6f}\n")

    models.append(model)

cv_mae = mean_absolute_error(y, oof)
print(f"{'='*60}")
print(f"🎯 CROSS-VALIDATION RESULTS")
print(f"{'='*60}")
print(f"  Fold MAEs:  {' | '.join([f'{m:.6f}' for m in fold_maes])}")
print(f"  CV MAE:     {cv_mae:.6f}")
print(f"{'='*60}\n")

# Ensemble prediction
print("🎲 Generating ensemble predictions...")
pred_eval = np.mean([m.predict(X_eval) for m in models], axis=0)

# Save submission
sub = pd.DataFrame({
    "data_id": eval_df["data_id"],
    "construction_cost_per_m2_usd": pred_eval
})

sub.to_csv(BASE / "submission_cnn_hybrid.csv", index=False)
print(f"\n✅ Submission saved: submission_cnn_hybrid.csv")
print(f"   Shape: {sub.shape}")
print(f"   Predictions range: [{pred_eval.min():.2f}, {pred_eval.max():.2f}]\n")
