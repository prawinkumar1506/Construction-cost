"""Merge ensemble predictions with evaluation mapping and write final_submission.csv

Usage:
    python scripts/make_final_submission.py

This script reads `submission_ensemble.csv` and
`datasets/evaluation_tabular_no_target.csv`, normalizes join keys,
merges predictions onto evaluation rows (by geolocation_name, year, quarter_label),
applies simple fallbacks for missing predictions, and writes
`final_submission.csv` in the project root.
"""
from pathlib import Path
import pandas as pd


BASE = Path(r"d:/Projects/Construction-cost")
ENS_PATH = BASE / "submission_ensemble.csv"
EVAL_PATH = BASE / "datasets" / "evaluation_tabular_no_target.csv"
OUT_PATH = BASE / "final_submission.csv"


def normalize(df):
    df = df.copy()
    if 'geolocation_name' in df.columns:
        df['geolocation_name'] = df['geolocation_name'].astype(str).str.strip()
    if 'quarter_label' in df.columns:
        df['quarter_label'] = df['quarter_label'].astype(str).str.strip()
    if 'year' in df.columns:
        df['year'] = df['year'].astype(str).str.strip()
    return df


def main():
    ens = pd.read_csv(ENS_PATH)
    eval_df = pd.read_csv(EVAL_PATH)

    ens = normalize(ens)
    eval_df = normalize(eval_df)

    # Primary merge: by geolocation_name, year, quarter_label
    merged = eval_df.merge(
        ens,
        on=['geolocation_name', 'year', 'quarter_label'],
        how='left',
        validate='1:1'
    )

    missing = merged['construction_cost_per_m2_usd'].isna().sum()
    print(f"After primary merge: {missing} missing predictions")

    if missing:
        # Fallback 1: merge by (geolocation_name, quarter_label) using median across years
        fallback = ens.groupby(['geolocation_name', 'quarter_label'], as_index=False)['construction_cost_per_m2_usd'].median()
        merged = merged.merge(
            fallback,
            on=['geolocation_name', 'quarter_label'],
            how='left',
            suffixes=('', '_fb')
        )

        # Use fallback values where primary predictions are missing
        merged['construction_cost_per_m2_usd'] = merged['construction_cost_per_m2_usd'].fillna(merged['construction_cost_per_m2_usd_fb'])
        merged = merged.drop(columns=[c for c in merged.columns if c.endswith('_fb')])
        missing = merged['construction_cost_per_m2_usd'].isna().sum()
        print(f"After fallback (geolocation+quarter median): {missing} missing predictions")

    if missing:
        # Fallback 2: fill remaining missing with overall median prediction
        overall_median = ens['construction_cost_per_m2_usd'].median()
        merged['construction_cost_per_m2_usd'] = merged['construction_cost_per_m2_usd'].fillna(overall_median)
        missing = merged['construction_cost_per_m2_usd'].isna().sum()
        print(f"After overall-median fallback: {missing} missing predictions (should be 0)")

    submission = merged[['data_id', 'construction_cost_per_m2_usd']].copy()
    submission.to_csv(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH} with {len(submission)} rows.")


if __name__ == '__main__':
    main()
