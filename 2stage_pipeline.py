# ================================================
# FINAL PIPELINE: CLIP + TABULAR + 2-STAGE MODEL
# Target: ~0.16–0.18 LB
# ================================================

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

# ================================================
# CONFIG
# ================================================
TARGET = "construction_cost_per_m2_usd"
N_FOLDS = 5
RANDOM_STATE = 42

TRAIN_CSV = r"datasets\train_tabular.csv"
TEST_CSV  = r"datasets\evaluation_tabular_no_target.csv"
CLIP_DIR  = r"cache\clip_feats"   # contains {data_id}.npy

# ================================================
# 1. LOAD DATA
# ================================================
print("🚀 Loading data...")

train = pd.read_csv(TRAIN_CSV)
test  = pd.read_csv(TEST_CSV)

print("Train:", train.shape)
print("Test :", test.shape)

# ================================================
# 2. LOAD CLIP FEATURES
# ================================================
print("📦 Loading CLIP embeddings...")

def load_clip(df):
    feats = []
    missing = 0
    for did in df["data_id"].values:
        path = os.path.join(CLIP_DIR, f"{did}.npy")
        if not os.path.exists(path):
            feats.append(np.zeros(512, dtype=np.float32))
            missing += 1
        else:
            feats.append(np.load(path))
    print("Missing CLIP:", missing)
    return np.vstack(feats)

clip_train = load_clip(train)
clip_test  = load_clip(test)

print("CLIP train shape:", clip_train.shape)
print("CLIP test shape :", clip_test.shape)

# ================================================
# 3. PCA COMPRESS CLIP
# ================================================
print("🧠 PCA compressing CLIP (512 → 64)...")

scaler = StandardScaler()
clip_train = scaler.fit_transform(clip_train)
clip_test  = scaler.transform(clip_test)

pca = PCA(n_components=64, random_state=42)
clip_train_pca = pca.fit_transform(clip_train)
clip_test_pca  = pca.transform(clip_test)

print("Explained variance:", pca.explained_variance_ratio_.sum())

# ================================================
# 4. TABULAR FEATURE ENGINEERING
# ================================================
print("🔧 Feature engineering...")

# Drop leaky columns
leaky = ["deflated_gdp_usd", "us_cpi"]
train = train.drop(columns=leaky, errors="ignore")
test  = test.drop(columns=leaky, errors="ignore")

# Temporal
train["year"] = train["quarter_label"].str[:4].astype(int)
test["year"]  = test["quarter_label"].str[:4].astype(int)

train["quarter"] = train["quarter_label"].str[-2:].map({"Q1":1,"Q2":2,"Q3":3,"Q4":4}).astype(int)
test["quarter"]  = test["quarter_label"].str[-2:].map({"Q1":1,"Q2":2,"Q3":3,"Q4":4}).astype(int)

# Country flag
train["is_japan"] = (train["country"] == "Japan").astype(int)
test["is_japan"]  = (test["country"] == "Japan").astype(int)

# Fill missing categorical
for c in train.columns:
    if train[c].dtype == "object":
        train[c] = train[c].fillna("unknown")
        test[c]  = test[c].fillna("unknown")

# Encode categoricals
exclude = {
    TARGET, "data_id", "geolocation_name", "quarter_label",
    "sentinel2_tiff_file_name", "viirs_tiff_file_name"
}

cat_cols = [c for c in train.columns if train[c].dtype == "object" and c not in exclude]

for c in cat_cols:
    le = LabelEncoder()
    train[c] = le.fit_transform(train[c])
    test[c]  = le.transform(test[c])

# Numeric columns
num_cols = [c for c in train.columns if c not in exclude and c not in cat_cols]

features = num_cols + cat_cols

X_tab = train[features].copy()
X_test_tab = test[features].copy()

X_tab = X_tab.fillna(X_tab.median())
X_test_tab = X_test_tab.fillna(X_tab.median())

# ================================================
# 5. CONCAT TABULAR + CLIP
# ================================================
print("🔗 Merging tabular + CLIP PCA...")

X = np.hstack([X_tab.values, clip_train_pca])
X_test = np.hstack([X_test_tab.values, clip_test_pca])

y = train[TARGET].values
groups = train["geolocation_name"].values

print("Final train shape:", X.shape)
print("Final test  shape:", X_test.shape)

# ================================================
# 6. TWO-STAGE TRAINING FUNCTION
# ================================================
def train_country(country_flag):
    idx = train["is_japan"].values == country_flag
    Xc = X[idx]
    yc = y[idx]
    gc = groups[idx]

    print("\n🌍 Training country:", "Japan" if country_flag==1 else "Philippines")
    print("Samples:", len(yc))

    base_mean = yc.mean()
    residual = yc - base_mean

    gkf = GroupKFold(n_splits=5)
    oof = np.zeros(len(yc))

    models = []

    for fold, (tr, va) in enumerate(gkf.split(Xc, residual, groups=gc)):
        print(f" Fold {fold+1}")

        model = lgb.LGBMRegressor(
            n_estimators=3000,
            learning_rate=0.03,
            num_leaves=64,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42+fold
        )

        model.fit(
            Xc[tr], residual[tr],
            eval_set=[(Xc[va], residual[va])],
            eval_metric="l1"
        )

        preds = model.predict(Xc[va])
        oof[va] = preds

        print("  Fold MAE:", mean_absolute_error(residual[va], preds))

        models.append(model)

    print(" Country residual CV MAE:", mean_absolute_error(residual, oof))

    return models, base_mean

# ================================================
# 7. TRAIN BOTH COUNTRIES
# ================================================
models_jp, mean_jp = train_country(1)
models_ph, mean_ph = train_country(0)

# ================================================
# 8. PREDICT TEST
# ================================================
print("\n🎯 Generating predictions...")

preds = np.zeros(len(test))

for i in range(len(test)):
    if test.loc[i, "is_japan"] == 1:
        p = np.mean([m.predict(X_test[i:i+1])[0] for m in models_jp])
        preds[i] = mean_jp + p
    else:
        p = np.mean([m.predict(X_test[i:i+1])[0] for m in models_ph])
        preds[i] = mean_ph + p

# ================================================
# 9. SAVE SUBMISSION
# ================================================
sub = pd.DataFrame({
    "data_id": test["data_id"],
    TARGET: preds
})

sub.to_csv("submission_clip_2stage.csv", index=False)

print("\n✅ Saved: submission_clip_2stage.csv")
print("Range:", preds.min(), preds.max())
print("Mean :", preds.mean())
