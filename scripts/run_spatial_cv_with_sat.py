import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

BASE = Path('d:/Projects/Construction-cost')
DATA_DIR = BASE / 'datasets'
CACHE_DIR = BASE / 'cache' / 'sat_v1'

from extract_sat_features import extract_features_for_pair

TRAIN_CSV = DATA_DIR / 'train_tabular.csv'
EVAL_CSV = DATA_DIR / 'evaluation_tabular_no_target.csv'

def build_sat_feature_table(df):
    # per-row feature table (with lat/lon extracted from raster bounds for true rings)
    df = df.copy()
    rows = []
    from tqdm import tqdm
    for idx, r in tqdm(df.iterrows(), total=len(df)):
        s = r.get('sentinel2_tiff_file_name')
        v = r.get('viirs_tiff_file_name')
        
        # extract center point from raster bounds
        lat, lon = None, None
        if s and pd.notna(s):
            try:
                import rasterio as rio
                rpath = DATA_DIR / s
                if rpath.exists():
                    with rio.open(rpath) as ds:
                        bounds = ds.bounds
                        lon = (bounds.left + bounds.right) / 2
                        lat = (bounds.top + bounds.bottom) / 2
            except Exception:
                pass
        
        feats = extract_features_for_pair(s, v, lat=lat, lon=lon)
        feats['row_id'] = idx
        rows.append(feats)
    sat_df = pd.DataFrame(rows)
    return sat_df

def main():
    train = pd.read_csv(TRAIN_CSV)
    eval_df = pd.read_csv(EVAL_CSV)

    # ---- Preprocessing (same as training pipeline) ----
    train['tropical_cyclone_wind_risk'] = train['tropical_cyclone_wind_risk'].fillna('Low')
    eval_df['tropical_cyclone_wind_risk'] = eval_df['tropical_cyclone_wind_risk'].fillna('Low')

    bool_cols = ['developed_country', 'landlocked', 'access_to_airport', 'access_to_port', 
                 'access_to_highway', 'access_to_railway', 'flood_risk_class']
    for col in bool_cols:
        if col in train.columns:
            train[col] = (train[col] == 'Yes').astype(int)
        if col in eval_df.columns:
            eval_df[col] = (eval_df[col] == 'Yes').astype(int)

    risk_map = {'Very Low': 0, 'Low': 1, 'Moderate': 2, 'High': 3, 'Very High': 4}
    for col in ['seismic_hazard_zone', 'tropical_cyclone_wind_risk', 'tornadoes_wind_risk']:
        if col in train.columns:
            train[col] = train[col].map(risk_map)
        if col in eval_df.columns:
            eval_df[col] = eval_df[col].map(risk_map)

    # preserve raw strings
    if 'country' in train.columns:
        train['country_str'] = train['country'].astype(str)
    if 'region_economic_classification' in train.columns:
        train['region_str'] = train['region_economic_classification'].astype(str)
    if 'koppen_climate_zone' in train.columns:
        train['koppen_str'] = train['koppen_climate_zone'].astype(str)
    if 'country' in eval_df.columns:
        eval_df['country_str'] = eval_df['country'].astype(str)
    if 'region_economic_classification' in eval_df.columns:
        eval_df['region_str'] = eval_df['region_economic_classification'].astype(str)
    if 'koppen_climate_zone' in eval_df.columns:
        eval_df['koppen_str'] = eval_df['koppen_climate_zone'].astype(str)

    # numeric encodings
    for col, enc_col in [
        ('country', 'country_enc'),
        ('region_economic_classification', 'region_enc'),
        ('koppen_climate_zone', 'koppen_enc')
    ]:
        if col in train.columns:
            le = LabelEncoder()
            train[enc_col] = le.fit_transform(train[col].astype(str))
            mapping = {k: v for v, k in enumerate(le.classes_)}
            if col in eval_df.columns:
                eval_df[enc_col] = eval_df[col].map(mapping).fillna(-1).astype(int)

    # time features
    def parse_quarter(q):
        try:
            year, qpart = q.split('-Q')
            return int(year), int(qpart)
        except Exception:
            return (np.nan, np.nan)

    train[['year_from_q', 'quarter']] = train['quarter_label'].apply(lambda x: pd.Series(parse_quarter(x)))
    eval_df[['year_from_q', 'quarter']] = eval_df['quarter_label'].apply(lambda x: pd.Series(parse_quarter(x)))

    if 'year' in train.columns:
        train['year_num'] = train['year'].astype(int)
    else:
        train['year_num'] = train['year_from_q']
    if 'year' in eval_df.columns:
        eval_df['year_num'] = eval_df['year'].astype(int)
    else:
        eval_df['year_num'] = eval_df['year_from_q']

    train['t'] = train['year_num'] * 4 + (train['quarter'] - 1)
    eval_df['t'] = eval_df['year_num'] * 4 + (eval_df['quarter'] - 1)

    t_min, t_max = train['t'].min(), train['t'].max()
    train['t_norm'] = (train['t'] - t_min) / (t_max - t_min + 1e-9)
    eval_df['t_norm'] = (eval_df['t'] - t_min) / (t_max - t_min + 1e-9)
    train['sin_q'] = np.sin(2 * np.pi * train['quarter'] / 4)
    train['cos_q'] = np.cos(2 * np.pi * train['quarter'] / 4)
    eval_df['sin_q'] = np.sin(2 * np.pi * eval_df['quarter'] / 4)
    eval_df['cos_q'] = np.cos(2 * np.pi * eval_df['quarter'] / 4)

    combined = pd.concat([train, eval_df])

    print('Building satellite feature table for', len(combined), 'locations...')
    sat_table = build_sat_feature_table(combined)
    sat_table.to_csv(BASE / 'sat_features_v1.csv', index=False)
    print('Saved sat_features_v1.csv with', len(sat_table), 'rows')

    # merge back by row index
    train = train.reset_index(drop=True)
    eval_df = eval_df.reset_index(drop=True)
    
    # extract row_ids for train/eval
    n_train = len(train)
    sat_train = sat_table[sat_table['row_id'] < n_train].set_index('row_id')
    sat_eval = sat_table[sat_table['row_id'] >= n_train].set_index('row_id')
    
    # shift eval row_ids back
    sat_eval.index = sat_eval.index - n_train
    
    train = train.join(sat_train)
    eval_df = eval_df.join(sat_eval)

    # fill numeric na (only where column exists in each frame)
    for c in train.columns:
        if train[c].dtype.kind in 'fc':
            train[c] = train[c].fillna(0)
            if c in eval_df.columns:
                eval_df[c] = eval_df[c].fillna(0)

    # prepare features (keep previous engineered set + sat features)
    base_feats = ['developed_country', 'landlocked', 'access_to_airport', 'access_to_port',
             'access_to_highway', 'access_to_railway', 'flood_risk_class',
             'seismic_hazard_zone', 'tropical_cyclone_wind_risk', 'tornadoes_wind_risk',
             'country_enc', 'region_enc', 'koppen_enc',
             't', 't_norm', 'sin_q', 'cos_q']

    sat_feats = [c for c in sat_table.columns if c not in ['tile_key', 'row_id']]
    feature_cols = [c for c in base_feats if c in train.columns] + sat_feats

    X = train[feature_cols].fillna(0)
    y = train['construction_cost_per_m2_usd']

    print('Running GroupKFold spatial CV with features:', len(feature_cols))
    gkf = GroupKFold(n_splits=5)
    lgb_models, xgb_models, cat_models = [], [], []
    maes = []
    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y, groups=train['geolocation_name'])):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        lgbm = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.03, num_leaves=127)
        lgbm.fit(X_tr.select_dtypes(include=[np.number]), y_tr,
                 eval_set=[(X_val.select_dtypes(include=[np.number]), y_val)],
                 eval_metric='l1', callbacks=[lgb.early_stopping(50)])
        lgb_pred = lgbm.predict(X_val.select_dtypes(include=[np.number]))

        xg = xgb.XGBRegressor(n_estimators=800, learning_rate=0.03)
        xg.fit(X_tr.select_dtypes(include=[np.number]), y_tr)
        xgb_pred = xg.predict(X_val.select_dtypes(include=[np.number]))

        # CatBoost (fast)
        X_tr_cb = X_tr.copy(); X_val_cb = X_val.copy()
        cat = CatBoostRegressor(iterations=300, learning_rate=0.05, depth=7, loss_function='MAE', random_seed=42+fold, verbose=0)
        cat.fit(X_tr_cb, y_tr, eval_set=(X_val_cb, y_val), early_stopping_rounds=50, verbose=False)
        cat_pred = cat.predict(X_val_cb)

        ens = 0.4 * lgb_pred + 0.3 * xgb_pred + 0.3 * cat_pred
        mae = mean_absolute_error(y_val, ens)
        maes.append(mae)
        lgb_models.append(lgbm); xgb_models.append(xg); cat_models.append(cat)
        print(f'Fold {fold+1} MAE: {mae:.2f}')

    print(f'Overall spatial CV MAE: {np.mean(maes):.2f} ± {np.std(maes):.2f}')

if __name__ == '__main__':
    main()
