import pandas as pd
import os

EXT = "./external"

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 200)

def banner(title):
    print("\n" + "="*120)
    print(title)
    print("="*120)

def inspect_df(df, name):
    print(f"\n📄 {name}")
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("Dtypes:\n", df.dtypes)
    print("\nHead:")
    print(df.head())
    print("\nTail:")
    print(df.tail())
    print("\nMissing values (top 10):")
    print(df.isna().sum().sort_values(ascending=False).head(10))
    print("\nDescribe:")
    try:
        print(df.describe(include="all").T)
    except:
        print("Describe failed.")


# -----------------------------------
# 1. COMMODITY PRICES (KAGGLE CSV)
# -----------------------------------
banner("1. WORLD BANK COMMODITY PRICES (KAGGLE)")

comm_path = os.path.join(EXT, "commodity_prices.csv")
comm = pd.read_csv(comm_path)

inspect_df(comm, "commodity_prices.csv")

# Clean
if "Unnamed: 0" in comm.columns:
    comm = comm.drop(columns=["Unnamed: 0"])

comm["date"] = pd.to_datetime(comm["date"])
comm["year"] = comm["date"].dt.year

for c in comm.columns:
    if c not in ["date", "year"]:
        comm[c] = pd.to_numeric(comm[c], errors="coerce")

comm_yearly = comm.groupby("year").mean(numeric_only=True).reset_index()

banner("Commodity Yearly Aggregated")

inspect_df(comm_yearly, "comm_yearly")


# -----------------------------------
# 2. ILO LABOUR PARTICIPATION
# -----------------------------------
banner("2. ILO LABOUR PARTICIPATION")

ilo_path = os.path.join(EXT, "EAP_2WAP_SEX_AGE_RT_A-20260126T1053.csv")
ilo = pd.read_csv(ilo_path, low_memory=False)

inspect_df(ilo, "ILO RAW")

# Filter
ilo_f = ilo[
    (ilo["sex.label"] == "Total") &
    (ilo["classif1.label"].str.contains("15\\+|Total", regex=True)) &
    (ilo["source.label"].str.contains("Modelled")) &
    (ilo["indicator.label"].str.contains("Labour force participation", case=False))
]

ilo_feat = ilo_f[["ref_area.label", "time", "obs_value"]].rename(columns={
    "ref_area.label": "country",
    "time": "year",
    "obs_value": "labour_participation_rate"
})

banner("ILO Labour Participation (Filtered)")

inspect_df(ilo_feat, "ilo_labour_participation")

print("\nCountry count:", ilo_feat["country"].nunique())
print("Year range:", ilo_feat["year"].min(), "→", ilo_feat["year"].max())


# -----------------------------------
# 3. ILO UNEMPLOYMENT
# -----------------------------------
banner("3. ILO UNEMPLOYMENT")

une_path = os.path.join(EXT, "UNE_2EAP_SEX_AGE_RT_A-20260126T1053.csv")
une = pd.read_csv(une_path, engine="python", on_bad_lines="skip")

inspect_df(une, "UNE RAW")

# Filter
une_f = une[
    (une["sex.label"] == "Total") &
    (une["classif1.label"].str.contains("15\\+|Total", regex=True)) &
    (une["source.label"].str.contains("Modelled")) &
    (une["indicator.label"].str.contains("Unemployment", case=False))
]

une_feat = une_f[["ref_area.label", "time", "obs_value"]].rename(columns={
    "ref_area.label": "country",
    "time": "year",
    "obs_value": "unemployment_rate"
})

banner("ILO Unemployment (Filtered)")

inspect_df(une_feat, "ilo_unemployment")

print("\nCountry count:", une_feat["country"].nunique())
print("Year range:", une_feat["year"].min(), "→", une_feat["year"].max())


# -----------------------------------
# 4. WORLD BANK GDP DEFLATOR
# -----------------------------------
banner("4. WORLD BANK GDP DEFLATOR")

gdp_file = [f for f in os.listdir(EXT) if f.startswith("API_NY.GDP.DEFL")][0]
gdp_path = os.path.join(EXT, gdp_file)

gdp_raw = pd.read_csv(gdp_path, skiprows=4)

inspect_df(gdp_raw, "GDP RAW")

# Melt to long
year_cols = [c for c in gdp_raw.columns if c.isdigit()]

gdp_long = gdp_raw.melt(
    id_vars=["Country Name"],
    value_vars=year_cols,
    var_name="year",
    value_name="gdp_deflator"
)

gdp_long["year"] = gdp_long["year"].astype(int)

gdp_feat = gdp_long.rename(columns={
    "Country Name": "country"
})

banner("GDP Deflator (Processed)")

inspect_df(gdp_feat, "gdp_deflator")

print("\nCountry count:", gdp_feat["country"].nunique())
print("Year range:", gdp_feat["year"].min(), "→", gdp_feat["year"].max())


# -----------------------------------
# 5. SUMMARY
# -----------------------------------
banner("5. SUMMARY")

print("Commodity yearly:", comm_yearly.shape)
print("ILO labour:", ilo_feat.shape)
print("ILO unemployment:", une_feat.shape)
print("GDP deflator:", gdp_feat.shape)

print("\nALL DATASETS LOADED SUCCESSFULLY ✅")
print("NEXT STEP: We will normalize country names and build the merge table.")
