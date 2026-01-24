import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from tqdm import tqdm

# ==============================
# CONFIG
# ==============================
TRAIN_CSV = "datasets/train_tabular.csv"
TEST_CSV  = "datasets/evaluation_tabular_no_target.csv"
CLIP_DIR  = "cache/clip_feats"

TARGET = "construction_cost_per_m2_usd"
N_FOLDS = 5
RANDOM_STATE = 42

# ==============================
# LOAD DATA
# ==============================
print("Loading data...")
train = pd.read_csv(TRAIN_CSV)
test  = pd.read_csv(TEST_CSV)

# ==============================
# LOAD CLIP FEATURES
# ==============================
def load_clip_features(df, clip_dir):
    feats = []
    missing = 0
    for did in df["data_id"]:
        path = os.path.join(clip_dir, f"{did}.npy")
        if not os.path.exists(path):
            feats.append(np.zeros(512, dtype=np.float32))
            missing += 1
        else:
            feats.append(np.load(path))
    print("Missing CLIP:", missing)
    return np.vstack(feats)

print("Loading CLIP features...")
clip_train = load_clip_features(train, CLIP_DIR)
clip_test  = load_clip_features(test, CLIP_DIR)

# ==============================
# BASIC FEATURE ENGINEERING
# ==============================
def add_time_features(df):
    df = df.copy()
    df["year"] = df["quarter_label"].str[:4].astype(int)
    df["quarter"] = df["quarter_label"].str[-1].astype(int)
    return df

train = add_time_features(train)
test  = add_time_features(test)

train["is_japan"] = (train["country"] == "Japan").astype(int)
test["is_japan"]  = (test["country"] == "Japan").astype(int)

# ==============================
# ENCODE CATEGORICALS
# ==============================
exclude = {
    TARGET, "data_id", "geolocation_name", "quarter_label",
    "sentinel2_tiff_file_name", "viirs_tiff_file_name"
}

cat_cols = [c for c in train.columns if train[c].dtype == "object" and c not in exclude]
num_cols = [c for c in train.columns if c not in exclude and c not in cat_cols]

for c in cat_cols:
    le = LabelEncoder()
    train[c] = le.fit_transform(train[c].astype(str))
    test[c]  = le.transform(test[c].astype(str))

# ==============================
# BASE TABULAR FEATURES
# ==============================
X_tab_train = train[num_cols + cat_cols].copy()
X_tab_test  = test[num_cols + cat_cols].copy()

X_tab_train = X_tab_train.fillna(X_tab_train.median())
X_tab_test  = X_tab_test.fillna(X_tab_train.median())

# ==============================
# CONCAT CLIP
# ==============================
X_train_full = np.hstack([X_tab_train.values, clip_train])
X_test_full  = np.hstack([X_tab_test.values,  clip_test])

# ==============================
# TWO STAGE PER COUNTRY
# ==============================
test_preds = np.zeros(len(test))

for country_flag, country_name in [(1, "Japan"), (0, "Philippines")]:
    print(f"\n==============================")
    print(f"Training country: {country_name}")
    print(f"==============================")

    idx_tr = train["is_japan"].values == country_flag
    idx_te = test["is_japan"].values == country_flag

    Xc = X_train_full[idx_tr]
    yc = train.loc[idx_tr, TARGET].values
    groups = train.loc[idx_tr, "geolocation_name"].values

    Xc_test = X_test_full[idx_te]

    # If only 1 group exists, train single model
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        print("⚠️ Only one location, training single model")
        model = lgb.LGBMRegressor(
            n_estimators=2000,
            learning_rate=0.03,
            num_leaves=128,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE
        )
        model.fit(Xc, yc)
        test_preds[idx_te] = model.predict(Xc_test)
        continue

    gkf = GroupKFold(n_splits=min(N_FOLDS, len(unique_groups)))
    oof = np.zeros(len(yc))

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(Xc, yc, groups)):
        Xtr, Xva = Xc[tr_idx], Xc[va_idx]
        ytr, yva = yc[tr_idx], yc[va_idx]

        model = lgb.LGBMRegressor(
            n_estimators=5000,
            learning_rate=0.03,
            num_leaves=128,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=10,
            random_state=RANDOM_STATE + fold
        )

        model.fit(
            Xtr, ytr,
            eval_set=[(Xva, yva)],
            eval_metric="l1",
            callbacks=[lgb.early_stopping(200, verbose=False)]
        )

        oof[va_idx] = model.predict(Xva)

    print(" Country CV MAE:", mean_absolute_error(yc, oof))

    # Train final model
    final_model = lgb.LGBMRegressor(
        n_estimators=5000,
        learning_rate=0.03,
        num_leaves=128,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=10,
        random_state=RANDOM_STATE
    )

    final_model.fit(Xc, yc)
    test_preds[idx_te] = final_model.predict(Xc_test)

# ==============================
# SAVE SUBMISSION (SAFE)
# ==============================
raw_test = pd.read_csv(TEST_CSV)

sub = pd.DataFrame({
    "data_id": raw_test["data_id"].astype(str).values,
    TARGET: test_preds
})

sub.to_csv("submission_final_clip_location.csv", index=False)

print("\nSaved: submission_final_clip_location.csv")
print(sub.head())
print("Rows:", len(sub))
print("Range:", sub[TARGET].min(), sub[TARGET].max())
print("Mean :", sub[TARGET].mean())
