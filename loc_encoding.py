import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error

# ================================
# CONFIG
# ================================
TRAIN_PATH = "datasets/train_tabular.csv"
TEST_PATH  = "datasets/evaluation_tabular_no_target.csv"
OUT_FILE   = "submission_final_012.csv"

TARGET = "construction_cost_per_m2_usd"
N_FOLDS = 5
SEED = 42

# ================================
# LOAD
# ================================
print("📂 Loading data...")
train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)

train["is_train"] = 1
test["is_train"] = 0
test[TARGET] = np.nan

df = pd.concat([train, test], ignore_index=True)

# ================================
# TIME FEATURES
# ================================
df["year"] = df["quarter_label"].str[:4].astype(int)
df["q"] = df["quarter_label"].str[-1].astype(int)
df["t"] = df["year"] * 4 + df["q"]

# ================================
# SORT FOR TIME SERIES
# ================================
df = df.sort_values(["geolocation_name", "t"]).reset_index(drop=True)

# ================================
# LOCATION MEMORY FEATURES
# ================================
print("🧠 Building time-series features...")

grp = df.groupby("geolocation_name")

df["lag1"] = grp[TARGET].shift(1)
df["lag2"] = grp[TARGET].shift(2)
df["lag3"] = grp[TARGET].shift(3)

df["vel1"] = df["lag1"] - df["lag2"]
df["vel2"] = df["lag2"] - df["lag3"]
df["acc1"] = df["vel1"] - df["vel2"]

df["mean_loc"] = grp[TARGET].transform("mean")
df["std_loc"]  = grp[TARGET].transform("std")

df["last_known"] = df["lag1"]

# ================================
# COUNTRY BAYESIAN MEAN
# ================================
print("🌍 Building country Bayesian priors...")

train_only = df[df["is_train"] == 1]

country_mean = train_only.groupby("country")[TARGET].mean()
global_mean = train_only[TARGET].mean()

df["country_mean"] = df["country"].map(country_mean)
df["country_mean"] = df["country_mean"].fillna(global_mean)

# Bayesian shrinkage
df["bayes_mean"] = (df["mean_loc"] * 5 + df["country_mean"] * 2) / 7

# ================================
# SEASONALITY
# ================================
print("🗓️ Building seasonality features...")

season_mean = (
    train_only
    .groupby(["country","q"])[TARGET]
    .mean()
    .reset_index()
)

season_mean.columns = ["country","q","season_mean"]

df = df.merge(season_mean, on=["country","q"], how="left")
df["season_mean"] = df["season_mean"].fillna(global_mean)

df["season_adj"] = df["season_mean"] - df["country_mean"]

# ================================
# FILL NA
# ================================
for c in ["lag1","lag2","lag3","vel1","vel2","acc1","mean_loc","std_loc","last_known"]:
    df[c] = df[c].fillna(df["bayes_mean"])

# ================================
# SPLIT BACK
# ================================
train_df = df[df["is_train"]==1].copy()
test_df  = df[df["is_train"]==0].copy()

# ================================
# BASELINE MODEL
# ================================
train_df["baseline"] = (
    0.6 * train_df["last_known"] +
    0.3 * train_df["bayes_mean"] +
    0.1 * train_df["season_mean"]
)

train_df["residual"] = train_df[TARGET] - train_df["baseline"]

# ================================
# MODEL FEATURES
# ================================
features = [
    "lag1","lag2","lag3",
    "vel1","vel2","acc1",
    "mean_loc","std_loc",
    "bayes_mean","season_adj"
]

X = train_df[features]
y = train_df["residual"]
X_test = test_df[features]

groups = train_df["geolocation_name"]

# ================================
# TRAIN RESIDUAL MODEL
# ================================
print("🚀 Training residual model...")

params = dict(
    n_estimators=4000,
    learning_rate=0.015,
    num_leaves=128,
    max_depth=10,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=SEED
)

gkf = GroupKFold(n_splits=5)

oof = np.zeros(len(X))
test_pred = np.zeros(len(X_test))

fold_maes = []

for fold,(tr,va) in enumerate(gkf.split(X,y,groups)):
    print(f"\n🔁 Fold {fold+1}")

    model = lgb.LGBMRegressor(**params)

    model.fit(
        X.iloc[tr], y.iloc[tr],
        eval_set=[(X.iloc[va], y.iloc[va])],
        callbacks=[lgb.early_stopping(300, verbose=False)]
    )

    pred = model.predict(X.iloc[va])
    oof[va] = pred

    mae = mean_absolute_error(y.iloc[va], pred)
    fold_maes.append(mae)
    print("  Residual MAE:", mae)

    test_pred += model.predict(X_test) / N_FOLDS

print("\n==============================")
print("📊 Residual CV MAE:", mean_absolute_error(y, oof))
print("Per-fold:", fold_maes)
print("==============================")

# ================================
# FINAL PREDICTION
# ================================
final_test = (
    0.6 * test_df["last_known"].values +
    0.3 * test_df["bayes_mean"].values +
    0.1 * test_df["season_mean"].values +
    test_pred
)

final_test = np.clip(final_test, 100, 5000)

# ================================
# SAVE SUBMISSION
# ================================
sub = pd.DataFrame({
    "data_id": test_df["data_id"].astype(str).values,
    TARGET: final_test
})

sub.to_csv(OUT_FILE, index=False)

print("\n✅ Saved:", OUT_FILE)
print("Range:", final_test.min(), final_test.max())
print("Mean :", final_test.mean())
print("DONE")
