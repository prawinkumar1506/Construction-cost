"""
Final competition pipeline: tabular-first, country-separated, production-grade.
Objective: ~0.17-0.18 LB via strong tabular features + blending + post-processing.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

BASE = Path('d:/Projects/Construction-cost')
DATA_DIR = BASE / 'datasets'
TRAIN_CSV = DATA_DIR / 'train_tabular.csv'
EVAL_CSV = DATA_DIR / 'evaluation_tabular_no_target.csv'

# =========================
# Feature Engineering
# =========================

def build_features(df):
    df = df.copy()

    # Binary flags
    bool_cols = ['developed_country', 'landlocked', 'access_to_airport', 'access_to_port',
                 'access_to_highway', 'access_to_railway', 'flood_risk_class']
    for col in bool_cols:
        if col in df.columns:
            if df[col].dtype == 'object':
                df[col] = (df[col] == 'Yes').astype(int)
            else:
                df[col] = df[col].fillna(0).astype(int)

    # Distance transforms
    df['log_dist'] = np.log1p(df['straight_distance_to_capital_km'])
    df['inv_dist'] = 1.0 / (df['straight_distance_to_capital_km'] + 1.0)

    # Infrastructure score
    access_cols = ['access_to_airport', 'access_to_port', 'access_to_highway', 'access_to_railway']
    df['infra_score'] = df[[c for c in access_cols if c in df.columns]].sum(axis=1)

    # Risk score
    risk_map = {'Very Low': 0, 'Low': 1, 'Moderate': 2, 'High': 3, 'Very High': 4}
    risk_cols = ['seismic_hazard_zone', 'flood_risk_class', 'tropical_cyclone_wind_risk', 'tornadoes_wind_risk']
    df['risk_score'] = 0
    for col in risk_cols:
        if col in df.columns:
            df['risk_score'] += df[col].map(risk_map).fillna(1).astype(int)

    # GDP transforms
    df['gdp_log'] = np.log1p(df['deflated_gdp_usd'])
    df['gdp_x_infra'] = df['gdp_log'] * df['infra_score']
    df['gdp_per_km'] = df['gdp_log'] / (df['straight_distance_to_capital_km'] + 1.0)

    # Development interactions
    df['dev_x_infra'] = df['developed_country'] * df['infra_score']
    df['dev_x_risk'] = df['developed_country'] * df['risk_score']

    # Climate × risk
    if 'koppen_climate_zone' in df.columns:
        df['climate_encoded'] = pd.factorize(df['koppen_climate_zone'])[0]
        df['climate_x_risk'] = df['climate_encoded'] * df['risk_score']

    # Time features
    def parse_quarter(q):
        try:
            year, qpart = str(q).split('-Q')
            return int(year), int(qpart)
        except:
            return np.nan, np.nan

    df[['year_parsed', 'quarter_parsed']] = df['quarter_label'].apply(lambda x: pd.Series(parse_quarter(x)))
    df['t'] = df['year_parsed'] * 4 + (df['quarter_parsed'] - 1)

    # CPI transform
    if 'us_cpi' in df.columns:
        df['cpi_log'] = np.log1p(df['us_cpi'])

    return df


def create_feature_list(df):
    exclude = [
        'construction_cost_per_m2_usd',
        'geolocation_name',
        'data_id',
        'country',
        'year',
        'quarter_label',
        'target_resid',
        'sentinel2_tiff_file_name',
        'viirs_tiff_file_name',
        'region_economic_classification',
        'koppen_climate_zone',
        'quarter_parsed',
        'year_parsed'
    ]

    feats = [c for c in df.columns if c not in exclude and np.issubdtype(df[c].dtype, np.number)]
    return feats


# =========================
# Training per country
# =========================

def train_country_models(country_name, train_df, eval_df, feature_cols):
    print(f"\n{'='*60}")
    print(f"Training {country_name} models")
    print(f"{'='*60}")

    X = train_df[feature_cols].fillna(0)
    y = train_df['target_resid']
    groups = train_df['geolocation_name'].values

    gkf = GroupKFold(n_splits=5)

    oof_lgb = np.zeros(len(X))
    oof_xgb = np.zeros(len(X))
    oof_cat = np.zeros(len(X))

    lgb_models = []
    xgb_models = []
    cat_models = []

    fold_maes = []

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        # LightGBM
        lgbm = lgb.LGBMRegressor(
            n_estimators=2000,
            learning_rate=0.03,
            num_leaves=127,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        lgbm.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric='mae',
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(False)]
        )
        pred_lgb = lgbm.predict(X_val)
        oof_lgb[val_idx] = pred_lgb
        lgb_models.append(lgbm)

        # XGBoost
        xgbm = xgb.XGBRegressor(
            n_estimators=1500,
            learning_rate=0.03,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        xgbm.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        pred_xgb = xgbm.predict(X_val)
        oof_xgb[val_idx] = pred_xgb
        xgb_models.append(xgbm)

        # CatBoost
        cat = CatBoostRegressor(
            iterations=2000,
            learning_rate=0.03,
            depth=7,
            loss_function='MAE',
            random_seed=42,
            verbose=0
        )
        cat.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=100, verbose=False)
        pred_cat = cat.predict(X_val)
        oof_cat[val_idx] = pred_cat
        cat_models.append(cat)

        blend_pred = 0.3 * pred_lgb + 0.3 * pred_xgb + 0.4 * pred_cat
        mae = mean_absolute_error(y_val, blend_pred)
        fold_maes.append(mae)
        print(f"Fold {fold+1}: MAE = {mae:.2f}")

    print(f"Mean CV MAE ({country_name}): {np.mean(fold_maes):.2f} ± {np.std(fold_maes):.2f}")

    # Train blender
    stack_train = pd.DataFrame({
        'lgb': oof_lgb,
        'xgb': oof_xgb,
        'cat': oof_cat
    })

    blender = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])
    blender.fit(stack_train, y)
    print(f"Blender alpha: {blender.alpha_}")

    # Predict eval
    X_eval = eval_df[feature_cols].fillna(0)

    pred_eval_lgb = np.mean([m.predict(X_eval) for m in lgb_models], axis=0)
    pred_eval_xgb = np.mean([m.predict(X_eval) for m in xgb_models], axis=0)
    pred_eval_cat = np.mean([m.predict(X_eval) for m in cat_models], axis=0)

    stack_eval = pd.DataFrame({
        'lgb': pred_eval_lgb,
        'xgb': pred_eval_xgb,
        'cat': pred_eval_cat
    })

    pred_eval = blender.predict(stack_eval)

    return pred_eval, np.mean(fold_maes), np.std(fold_maes)


# =========================
# Main
# =========================

def main():
    print("Loading data...")
    train = pd.read_csv(TRAIN_CSV)
    eval_df = pd.read_csv(EVAL_CSV)

    print("Building features...")
    train = build_features(train)
    eval_df = build_features(eval_df)

    feature_cols = create_feature_list(train)
    print(f"Using {len(feature_cols)} features")

    countries = train['country'].unique()
    print(f"Countries: {countries}")

    all_preds = []

    for country in countries:
        train_c = train[train['country'] == country].copy()
        eval_c = eval_df[eval_df['country'] == country].copy()

        country_mean = train_c['construction_cost_per_m2_usd'].mean()
        train_c['target_resid'] = train_c['construction_cost_per_m2_usd'] - country_mean

        pred_eval, cv_mean, cv_std = train_country_models(country, train_c, eval_c, feature_cols)

        eval_c = eval_c.copy()
        eval_c['pred'] = pred_eval + country_mean
        all_preds.append(eval_c[['data_id', 'country', 'geolocation_name', 'pred']])

    print("\nMerging predictions...")
    final_preds = pd.concat(all_preds, ignore_index=True)
    eval_df = eval_df.merge(final_preds[['data_id', 'pred']], on='data_id', how='left')

    print("\nPost-processing...")

    # Clip per country
    clip_ranges = {
        'Philippines': (100, 900),
        'Japan': (800, 3000)
    }

    for country, (lo, hi) in clip_ranges.items():
        mask = eval_df['country'] == country
        eval_df.loc[mask, 'pred'] = eval_df.loc[mask, 'pred'].clip(lo, hi)

    # Time smoothing per (country, geolocation)
    for (country, geo), idx in eval_df.groupby(['country', 'geolocation_name']).groups.items():
        if len(idx) > 1:
            vals = eval_df.loc[idx, 'pred'].values
            smoothed = pd.Series(vals).rolling(window=2, center=True, min_periods=1).mean()
            eval_df.loc[idx, 'pred'] = smoothed.values

    # Save submission
    submission_df = pd.DataFrame({
        'data_id': eval_df['data_id'],
        'construction_cost_per_m2_usd': eval_df['pred']
    })

    out_path = BASE / 'final_submission_tabular.csv'
    submission_df.to_csv(out_path, index=False)
    print(f"\nSaved submission to {out_path}")


if __name__ == '__main__':
    main()
