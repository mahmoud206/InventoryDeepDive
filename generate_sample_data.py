"""
generate_sample_data.py  ·  project root
=========================================
Generates a realistic synthetic dataset that mirrors the schema of
combined_df exactly — same columns, same dtypes, same value ranges —
but contains zero real business data.

Use this file to run the notebooks in CI or for public demonstration.

Usage
-----
    python generate_sample_data.py
    # writes outputs/combined_df.parquet  (~40 MB for default config)
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Configuration ─────────────────────────────────────────────────────────────

N_SKUS         = 2_000      # unique SKUs
N_SECTIONS     = 80         # product sections (first 4 digits of SKU)
N_SUPPLIERS    = 30
N_OUTLETS      = 150
DATE_START     = "2019-01-01"
DATE_END       = "2025-12-31"
TRANSACTIONS   = 200_000    # total transaction rows
OUTPUT_DIR     = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
RANDOM_SEED    = 42

rng = np.random.default_rng(RANDOM_SEED)

# ── Reference tables ──────────────────────────────────────────────────────────

sections   = [str(rng.integers(3000, 6000)) for _ in range(N_SECTIONS)]
sections   = list(dict.fromkeys(sections))[:N_SECTIONS]

skus = []
for sec in sections:
    n = rng.integers(10, 60)
    for _ in range(n):
        color  = rng.integers(1, 30)
        suffix = rng.integers(10, 30)
        skus.append(f"{sec}{color:03d}{suffix:04d}")
skus = list(dict.fromkeys(skus))[:N_SKUS]

suppliers = [f"SUPPLIER_{i:03d}" for i in range(N_SUPPLIERS)]
outlet_names = [f"Outlet_{i:04d}" for i in range(N_OUTLETS)]
outlet_types = rng.choice(["Retail", "Wholesale", "Contract", "Online"], size=N_OUTLETS)
outlet_areas = rng.choice(["Central", "North", "South", "East", "West"], size=N_OUTLETS)
outlet_countries = rng.choice(["SA", "AE", "KW", "QA", "BH"], size=N_OUTLETS, p=[0.6, 0.2, 0.08, 0.07, 0.05])

outlet_df = pd.DataFrame({
    "Outlet Name":           outlet_names,
    "Outlet_Type":           outlet_types,
    "Area_Master_Category":  outlet_areas,
    "Country":               outlet_countries,
})

categories    = rng.choice(["A", "B", "C", "D"], size=N_SKUS, p=[0.25, 0.35, 0.25, 0.15])
sku_suppliers = rng.choice(suppliers, size=N_SKUS)
sku_stocks    = rng.integers(0, 500, size=N_SKUS).astype(np.int32)
sku_costs     = (rng.uniform(5, 80, size=N_SKUS)).round(2)

sku_meta = pd.DataFrame({
    "SKU":           skus,
    "CATEGORY":      categories,
    "Supplier":      sku_suppliers,
    "CURRENT_STOCK": sku_stocks,
    "Unit_Cost":     sku_costs,
})
sku_section_map = {sku: sku[:4] for sku in skus}

# ── Transaction generation ────────────────────────────────────────────────────

print(f"Generating {TRANSACTIONS:,} transactions  ·  {N_SKUS:,} SKUs  ·  {N_SECTIONS} sections …")

dates = pd.to_datetime(
    rng.integers(
        pd.Timestamp(DATE_START).value,
        pd.Timestamp(DATE_END).value,
        size=TRANSACTIONS,
    )
).normalize()

# Heavier weight on more active SKUs (Pareto-like)
sku_weights = np.exp(-rng.exponential(scale=1.5, size=N_SKUS))
sku_weights /= sku_weights.sum()
txn_skus = rng.choice(skus, size=TRANSACTIONS, p=sku_weights)

# Contract orders (bal Qty >= 100) ~ 8% of transactions
is_contract = rng.random(TRANSACTIONS) < 0.08
bal_qty = np.where(
    is_contract,
    rng.integers(100, 2000, size=TRANSACTIONS),
    rng.integers(1,   50,   size=TRANSACTIONS),
).astype(np.float32)

# Unit price — correlated with cost
txn_cost_idx = [skus.index(s) for s in txn_skus]
unit_costs   = sku_costs[txn_cost_idx]
unit_price   = (unit_costs * rng.uniform(1.2, 2.5, size=TRANSACTIONS)).astype(np.float32)
bal_value    = (bal_qty * unit_price).astype(np.float32)
total_cost   = (bal_qty * unit_costs).astype(np.float32)
total_profit = (bal_value - total_cost).astype(np.float32)

txn_outlets  = rng.choice(outlet_names, size=TRANSACTIONS)
outlet_lookup = outlet_df.set_index("Outlet Name")

df = pd.DataFrame({
    "Date":         dates,
    "SKU":          txn_skus,
    "Section":      [sku_section_map[s] for s in txn_skus],
    "bal Value":    bal_value,
    "bal Qty":      bal_qty,
    "U Price":      unit_price,
    "Outlet Name":  txn_outlets,
    "Outlet":       rng.integers(1000, 9999, size=TRANSACTIONS).astype(np.float32),
    "Client":       pd.array([f"C{rng.integers(1000,9999)}" for _ in range(TRANSACTIONS)], dtype="string"),
    "Year":         pd.array(dates.year, dtype="Int64"),
})

# Merge metadata
df = df.merge(sku_meta[["SKU", "CATEGORY", "Supplier", "CURRENT_STOCK"]], on="SKU", how="left")
df = df.merge(outlet_lookup[["Outlet_Type", "Area_Master_Category", "Country"]],
              left_on="Outlet Name", right_index=True, how="left")

df["Total_Cost"]      = total_cost
df["Total_Profit"]    = total_profit
df["Profit_Margin_%"] = np.where(bal_value > 0, total_profit / bal_value * 100, 0).astype(np.float32)
df["Sales_Type"]      = np.where(bal_qty >= 100, "CONTRACT", "FAMILY")

# Entry dates and first-invoice dates (section-level)
df["LAST_ENTRY_DATE"] = pd.to_datetime(
    df.groupby("SKU")["Date"].transform("max"))

first_inv = (df.groupby("Section")["Date"].min()
               .reset_index()
               .rename(columns={"Date": "First_Inv_Date"}))
df = df.merge(first_inv, on="Section", how="left")

# Nb days available balance (synthetic)
df["Nb. Days (Avail. Balance)"] = rng.uniform(0, 365, size=len(df)).astype(np.float32)

# Historical qty (pre-2019)
df["OLD_QTY"] = np.where(
    rng.random(len(df)) < 0.4,
    rng.uniform(10, 5000, size=len(df)).astype(np.float32),
    np.nan,
)

# Art pattern / texture / colour
df["ARTPATERN"]    = rng.choice(["PLAIN", "PLAIN", "STRIPE", "FLORAL", "GEOMETRIC"], size=len(df))
df["TEXTURE"]      = rng.choice(["BLACKOUT", "SHEER", "LINEN", "VELVET", "COTTON"],  size=len(df))
df["COLOR_NAME"]   = rng.choice(["WHITE", "BEIGE", "GREY", "NAVY", "OLIVE", "CREAM"], size=len(df))
df["IS_PLAIN_SECTION"] = np.where(
    df.groupby("Section")["ARTPATERN"].transform(lambda x: (x == "PLAIN").all()),
    "YES", "NO")

# Catalog / collection
df["CATALOG_NO"]      = rng.choice(
    [f"CAT-{i:03d}" for i in range(1, 41)], size=len(df))
df["COLLECTION_NAME"] = rng.choice(
    ["Spring", "Summer", "Autumn", "Winter", "Classic", "Premium"], size=len(df))
df["SERIAL"]          = rng.integers(1000, 9999, size=len(df)).astype(str)

df["COST_PER_DOLLAR"] = rng.uniform(3.5, 3.85, size=len(df)).astype(np.float32)

# Mahmoud classification (sample extra column)
df["Mahmoud_Class"] = rng.choice(["AA", "A", "B", "C", "D"], size=len(df),
                                   p=[0.05, 0.15, 0.30, 0.30, 0.20])

# Convert objects to StringDtype for Parquet compatibility
for col in df.select_dtypes("object").columns:
    df[col] = df[col].astype("string")

df = df.sort_values("Date").reset_index(drop=True)

# ── Save ──────────────────────────────────────────────────────────────────────

out = OUTPUT_DIR / "combined_df.parquet"
df.to_parquet(out, engine="pyarrow", compression="snappy", index=False)

size_mb = out.stat().st_size / 1024**2
print(f"\nSaved  →  {out}  ({size_mb:.1f} MB)")
print(f"Shape  :  {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print("\nThis file contains SYNTHETIC data only — safe for public repositories.")
