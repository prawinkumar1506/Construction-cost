import os
import pandas as pd
import numpy as np

EXTERNAL_DIR = "./external"

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 200)

def detect_columns(df):
    country_cols = [c for c in df.columns if any(x in c.lower() for x in ["country", "economy", "nation", "iso", "location"])]
    year_cols = [c for c in df.columns if any(x in c.lower() for x in ["year", "date", "time"])]
    value_cols = [c for c in df.columns if any(x in c.lower() for x in ["value", "rate", "index", "price", "usd"])]

    return country_cols, year_cols, value_cols


def inspect_df(df, name):
    if df.shape[0] == 0 or df.shape[1] == 0:
        print(f"\n⚠️ Skipping empty table: {name}")
        return

    print("\n" + "=" * 120)
    print(f"📄 FILE: {name}")
    print("=" * 120)

    print("\n🔹 Shape:", df.shape)
    print("\n🔹 Columns:")
    print(df.columns.tolist())

    print("\n🔹 Dtypes:")
    print(df.dtypes)

    print("\n🔹 Head:")
    print(df.head())

    print("\n🔹 Tail:")
    print(df.tail())

    print("\n🔹 Missing values:")
    print(df.isna().sum().sort_values(ascending=False).head(20))

    print("\n🔹 Numeric describe:")
    try:
        print(df.describe().T)
    except:
        print("No numeric columns to describe.")

    # Try detecting semantic columns
    country_cols, year_cols, value_cols = detect_columns(df)

    print("\n🧠 Auto-detected semantic columns:")
    print("  Country-like columns:", country_cols)
    print("  Year/Date-like columns:", year_cols)
    print("  Value-like columns:", value_cols)

    # If country column exists, show unique count
    if country_cols:
        c = country_cols[0]
        print(f"\n🌍 Unique {c} count:", df[c].nunique())
        print("Sample values:", df[c].dropna().unique()[:10])

    # If year column exists, show range
    if year_cols:
        y = year_cols[0]
        try:
            print(f"\n📅 {y} range:", df[y].min(), "→", df[y].max())
        except:
            pass

    print("\n🔹 Sample rows:")
    print(df.sample(min(5, len(df)), random_state=42))


def inspect_file(path):
    name = os.path.basename(path)

    if name.lower().endswith(".csv"):
        try:
            df = pd.read_csv(path)
        except:
            df = pd.read_csv(path, encoding="latin1")

        inspect_df(df, name)

    elif name.lower().endswith(".xlsx") or name.lower().endswith(".xls"):
        print("\n" + "=" * 120)
        print(f"📊 EXCEL FILE: {name}")
        print("=" * 120)

        xls = pd.ExcelFile(path)
        print("Sheets:", xls.sheet_names)

        for sheet in xls.sheet_names:
            df = pd.read_excel(path, sheet_name=sheet)
            inspect_df(df, f"{name} :: {sheet}")

    else:
        print(f"⚠️ Skipping unsupported file: {name}")


def main():
    print("🔍 Scanning directory:", EXTERNAL_DIR)
    files = os.listdir(EXTERNAL_DIR)

    print("\nFound files:")
    for f in files:
        print(" -", f)

    for f in files:
        path = os.path.join(EXTERNAL_DIR, f)
        if os.path.isfile(path):
            inspect_file(path)


if __name__ == "__main__":
    main()
