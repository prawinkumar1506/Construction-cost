import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

# ================================
# CONFIG
# ================================
TRAIN_CSV = "datasets/train_tabular.csv"
TEST_CSV = "datasets/evaluation_tabular_no_target.csv"
CLIP_DIR = "cache/clip_feats"
TARGET = "construction_cost_per_m2_usd"
N_FOLDS = 5
SEED = 42

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Device: {device}")

# ================================
# LOAD DATA
# ================================
print("📂 Loading data...")
train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)

train["is_train"] = 1
test["is_train"] = 0
test[TARGET] = np.nan

all_data = pd.concat([train, test], ignore_index=True)

# ================================
# TIME FEATURES
# ================================
all_data["year"] = all_data["quarter_label"].str[:4].astype(int)
all_data["quarter"] = all_data["quarter_label"].str[-1].astype(int)

all_data = all_data.sort_values(["geolocation_name", "year", "quarter"]).reset_index(drop=True)

# ================================
# LOCATION FEATURES
# ================================
print("🏗️ Building location features...")

grp = all_data.groupby("geolocation_name")

all_data["loc_mean"] = grp[TARGET].transform("mean")
all_data["loc_median"] = grp[TARGET].transform("median")
all_data["loc_std"] = grp[TARGET].transform("std")
all_data["loc_lag1"] = grp[TARGET].shift(1)
all_data["loc_lag2"] = grp[TARGET].shift(2)

country_grp = all_data.groupby("country")
all_data["country_mean"] = country_grp[TARGET].transform("mean")
all_data["country_std"] = country_grp[TARGET].transform("std")

# Fill NaNs
for c in ["loc_mean", "loc_median", "loc_std", "loc_lag1", "loc_lag2", "country_mean", "country_std"]:
    all_data[c] = all_data[c].fillna(all_data["country_mean"])

all_data = all_data.fillna(0)

# ================================
# LOAD CLIP
# ================================
print("🧠 Loading CLIP features...")
clip_feats = []
for did in all_data["data_id"].astype(str).values:
    path = f"{CLIP_DIR}/{did}.npy"
    if os.path.exists(path):
        clip_feats.append(np.load(path))
    else:
        clip_feats.append(np.zeros(512, dtype=np.float32))

clip_feats = np.vstack(clip_feats).astype('float32')

# ================================
# CLIP SIMILARITY (GPU)
# ================================
print("⚡ Computing CLIP KNN...")

train_mask = all_data["is_train"] == 1
clip_train = clip_feats[train_mask]
clip_test = clip_feats[~train_mask]

train_df = all_data[train_mask].reset_index(drop=True)
y_train_clip = train_df[TARGET].values

K = 10

clip_train_tensor = torch.from_numpy(clip_train).to(device)
clip_test_tensor = torch.from_numpy(clip_test).to(device)

clip_train_tensor = F.normalize(clip_train_tensor, p=2, dim=1)
clip_test_tensor = F.normalize(clip_test_tensor, p=2, dim=1)

batch_size = 256

# Train KNN
train_knn = []
for i in range(0, len(clip_train_tensor), batch_size):
    batch = clip_train_tensor[i:i+batch_size]
    sims = torch.mm(batch, clip_train_tensor.T)
    top_sims, top_idx = torch.topk(sims, k=K+1, dim=1)
    
    indices = top_idx[:, 1:].cpu().numpy()
    
    for j in range(len(indices)):
        neighbor_costs = y_train_clip[indices[j]]
        train_knn.append([
            np.mean(neighbor_costs),
            np.std(neighbor_costs),
            np.median(neighbor_costs)
        ])

train_knn = np.array(train_knn)

# Test KNN
test_knn = []
for i in range(0, len(clip_test_tensor), batch_size):
    batch = clip_test_tensor[i:i+batch_size]
    sims = torch.mm(batch, clip_train_tensor.T)
    top_sims, top_idx = torch.topk(sims, k=K, dim=1)
    
    indices = top_idx.cpu().numpy()
    
    for j in range(len(indices)):
        neighbor_costs = y_train_clip[indices[j]]
        test_knn.append([
            np.mean(neighbor_costs),
            np.std(neighbor_costs),
            np.median(neighbor_costs)
        ])

test_knn = np.array(test_knn)

del clip_train_tensor, clip_test_tensor
torch.cuda.empty_cache()

# Add to dataframe
all_data.loc[train_mask, 'clip_knn_mean'] = train_knn[:, 0]
all_data.loc[train_mask, 'clip_knn_std'] = train_knn[:, 1]
all_data.loc[train_mask, 'clip_knn_median'] = train_knn[:, 2]

all_data.loc[~train_mask, 'clip_knn_mean'] = test_knn[:, 0]
all_data.loc[~train_mask, 'clip_knn_std'] = test_knn[:, 1]
all_data.loc[~train_mask, 'clip_knn_median'] = test_knn[:, 2]

# ================================
# SPLIT
# ================================
train = all_data[all_data["is_train"] == 1].copy()
test = all_data[all_data["is_train"] == 0].copy()

# ================================
# PREPARE FEATURES
# ================================
exclude = {TARGET, "is_train", "data_id", "geolocation_name", "quarter_label", "sentinel2_tiff_file_name", "viirs_tiff_file_name"}

cat_cols = [c for c in train.columns if train[c].dtype == "object" and c not in exclude]

for c in cat_cols:
    le = LabelEncoder()
    train[c] = le.fit_transform(train[c].astype(str))
    test[c] = le.transform(test[c].astype(str))

features = [c for c in train.columns if c not in exclude]

X = train[features]
y = train[TARGET].values
X_test = test[features]
groups = train["geolocation_name"].values

print(f"Features: {len(features)}")

# ================================
# TRAIN ENSEMBLE
# ================================
gkf = GroupKFold(n_splits=N_FOLDS)

# LightGBM
print("\n📊 LightGBM...")
lgb_oof = np.zeros(len(X))
lgb_preds = np.zeros(len(X_test))

for fold, (tr, va) in enumerate(gkf.split(X, y, groups)):
    model = lgb.LGBMRegressor(
        n_estimators=3000,
        learning_rate=0.02,
        num_leaves=96,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED,
        n_jobs=-1,
        verbose=-1
    )
    model.fit(X.iloc[tr], y[tr])
    
    lgb_oof[va] = model.predict(X.iloc[va])
    lgb_preds += model.predict(X_test) / N_FOLDS
    print(f"  Fold {fold+1}: {mean_absolute_error(y[va], lgb_oof[va]):.4f}")

lgb_cv = mean_absolute_error(y, lgb_oof)
print(f"✅ LGB CV: {lgb_cv:.4f}")

# XGBoost
print("\n📊 XGBoost...")
xgb_oof = np.zeros(len(X))
xgb_preds = np.zeros(len(X_test))

for fold, (tr, va) in enumerate(gkf.split(X, y, groups)):
    model = xgb.XGBRegressor(
        n_estimators=2000,
        learning_rate=0.02,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED,
        tree_method='hist',
        n_jobs=-1
    )
    model.fit(X.iloc[tr], y[tr], verbose=False)
    
    xgb_oof[va] = model.predict(X.iloc[va])
    xgb_preds += model.predict(X_test) / N_FOLDS
    print(f"  Fold {fold+1}: {mean_absolute_error(y[va], xgb_oof[va]):.4f}")

xgb_cv = mean_absolute_error(y, xgb_oof)
print(f"✅ XGB CV: {xgb_cv:.4f}")

# CatBoost
print("\n📊 CatBoost...")
cat_oof = np.zeros(len(X))
cat_preds = np.zeros(len(X_test))

for fold, (tr, va) in enumerate(gkf.split(X, y, groups)):
    model = CatBoostRegressor(
        iterations=2000,
        depth=8,
        learning_rate=0.02,
        loss_function='MAE',
        random_seed=SEED,
        verbose=0
    )
    model.fit(X.iloc[tr], y[tr])
    
    cat_oof[va] = model.predict(X.iloc[va])
    cat_preds += model.predict(X_test) / N_FOLDS
    print(f"  Fold {fold+1}: {mean_absolute_error(y[va], cat_oof[va]):.4f}")

cat_cv = mean_absolute_error(y, cat_oof)
print(f"✅ CAT CV: {cat_cv:.4f}")

# ================================
# ENSEMBLE
# ================================
final_preds = 0.4 * lgb_preds + 0.35 * xgb_preds + 0.25 * cat_preds
final_preds = np.clip(final_preds, 50, 5000)

is_japan = test['country'] == 0  # 0 = Japan after encoding
final_preds[is_japan] = np.clip(final_preds[is_japan], 500, 5000)
final_preds[~is_japan] = np.clip(final_preds[~is_japan], 50, 1500)

# ================================
# SAVE
# ================================
sub = pd.DataFrame({
    "data_id": test["data_id"].values,
    TARGET: final_preds
})

sub.to_csv("submission_location_clip.csv", index=False)

print("\n" + "="*60)
print(f"✅ LGB: {lgb_cv:.4f} | XGB: {xgb_cv:.4f} | CAT: {cat_cv:.4f}")
print(f"✅ Saved: submission_location_clip.csv")
print(f"🎯 Expected LB: 0.15-0.17")
print("="*60)