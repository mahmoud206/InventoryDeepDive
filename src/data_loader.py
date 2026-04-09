"""
data_loader.py  ·  src/
========================
Sales data pipeline: ingest, enrich, and persist combined_df.parquet.

Usage
-----
    python src/data_loader.py

    from src.data_loader import run
    combined_df = run()
"""

import gc
import glob
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ── Configuration ─────────────────────────────────────────────────────────────

BASE_DIR    = Path(r"D:\Projects\InventoryDeepDive")
SALES_DIR   = BASE_DIR / "Sales_data"
HELPERS_DIR = BASE_DIR / "Helping_Files"
OUTPUT_DIR  = BASE_DIR / "Outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

FILES = {
    "yearly_costs"    : HELPERS_DIR / "SKU_COST_SHEET.xlsx",
    "fallback_cost"   : HELPERS_DIR / "SKUS_COST.xlsx",
    "stock"           : HELPERS_DIR / "current_stock.xlsx",
    "entry_date"      : HELPERS_DIR / "ENTRY_DATES.xlsx",
    "category"        : HELPERS_DIR / "SKU_CATEGORIES.xlsx",
    "supplier"        : HELPERS_DIR / "SUPPLIER_CODE.xlsx",
    "first_inv"       : HELPERS_DIR / "First Inv.date.xlsx",
    "mahmoud_class"   : HELPERS_DIR / "Mahmoud_SKU_Classification.xlsx",
    "nb_days"         : HELPERS_DIR / "nb.xlsx",
    "historical_qty"  : HELPERS_DIR / "Sales_Qty_1999_2018.xlsx",
    "artprint_texture": HELPERS_DIR / "ARTPRINT_TEXTURE.xlsx",
    "outlets"         : HELPERS_DIR / "Outlets.xlsx",
    "cost_per_dollar" : HELPERS_DIR / "COST_PER_DOLLAR.xlsx",
    "catalogs"        : HELPERS_DIR / "CATALOGS DATA.xlsx",
    "color"           : HELPERS_DIR / "color.xlsx",
}

EXCLUDED_OUTLETS = [
    "جزيرة", "مصنع الكتلوجات", "نجمة", "الادارة العامة التسويق", "الشامل"
]

DROP_COLS = [
    "Route", "Route Name", "cliroute", "Region", "Region Name", "Gen",
    "Credit Value", "Credit Qty", "Debit Value", "Debit Qty",
    "Zone", "zone name", "Salesman", "Salesman Name",
    "Ret Inv Nbr", "Price Cat", "Number", "Type", "Client Name",
    "درجة أهمية العميل ",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _clean_sku(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.replace(r"\.0+$", "", regex=True)


def _find_col(df: pd.DataFrame, keywords: list) -> str | None:
    for col in df.columns:
        if any(kw in col.lower() for kw in keywords):
            return col
    return None


def _dedup(df: pd.DataFrame, key: str) -> pd.DataFrame:
    return df.drop_duplicates(subset=[key], keep="first")


def _left_merge(base: pd.DataFrame, right: pd.DataFrame,
                on: str, label: str) -> pd.DataFrame:
    right = _dedup(right, on)
    n = len(base)
    merged = base.merge(right, on=on, how="left")
    if len(merged) != n:
        raise RuntimeError(
            f"[{label}] Left join inflated rows {n:,} → {len(merged):,}. "
            "Duplicate join keys in enrichment file."
        )
    return merged


# ── Stage 1 · Sales CSV ingestion ─────────────────────────────────────────────

def _load_sales(folder: Path, exclude_keywords: list) -> pd.DataFrame:
    csv_files = glob.glob(str(folder / "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files in {folder}")

    dtype_map = {
        "bal Value"  : "float32",
        "bal Qty"    : "float32",
        "U Price"    : "float32",
        "Client"     : "string",
        "Outlet"     : "float32",
        "Section"    : "str",
        "SKU"        : "str",
        "Outlet Name": "str",
    }

    chunks, n_in, n_out = [], 0, 0

    for path in csv_files:
        df = pd.read_csv(
            path,
            encoding="windows-1256",
            dtype=dtype_map,
            parse_dates=["Date"],
            dayfirst=True,
            low_memory=False,
        )
        n_in += len(df)

        if "Outlet Name" in df.columns:
            mask = pd.Series(False, index=df.index)
            for kw in exclude_keywords:
                mask |= df["Outlet Name"].str.contains(kw, case=False, na=False, regex=False)
            df = df[~mask]

        df["SKU"]  = _clean_sku(df["SKU"])
        df["Year"] = df["Date"].dt.year
        n_out += len(df)
        chunks.append(df)

        if len(chunks) >= 5:
            chunks = [pd.concat(chunks, ignore_index=True)]
            gc.collect()

    sales = pd.concat(chunks, ignore_index=True).sort_values("Date", ignore_index=True)
    del chunks
    gc.collect()

    print(f"  records : {n_out:,}  (excluded {n_in - n_out:,})")
    print(f"  range   : {sales['Date'].min().date()} → {sales['Date'].max().date()}")
    print(f"  skus    : {sales['SKU'].nunique():,}")
    return sales


# ── Stage 2 · Cost assignment & profit calculation ────────────────────────────

def _load_yearly_costs(path: Path) -> tuple:
    if not path.exists():
        return None, []

    df  = pd.read_excel(path)
    col = _find_col(df, ["sku"])
    if not col:
        return None, []

    df = df.rename(columns={col: "SKU"})
    df["SKU"] = _clean_sku(df["SKU"])

    year_re, cost_cols = re.compile(r"(19|20)\d{2}"), {}
    for c in df.columns:
        if c == "SKU":
            continue
        if any(kw in c.lower() for kw in ["cost", "price", "unit"]):
            m = year_re.search(c)
            if m:
                cost_cols[int(m.group())] = c

    if not cost_cols:
        return None, []

    result = df[["SKU"]].copy()
    for yr in sorted(cost_cols):
        result[f"Cost_{yr}"] = df[cost_cols[yr]].astype("float32")

    return result, sorted(cost_cols)


def _load_fallback_costs(path: Path) -> dict:
    if not path.exists():
        return {}
    df = pd.read_excel(path)
    sc = _find_col(df, ["sku"])
    cc = _find_col(df, ["cost"])
    if not sc or not cc:
        return {}
    df["SKU"] = _clean_sku(df[sc])
    return df.set_index("SKU")[cc].astype(float).to_dict()


def _calculate_profits(df: pd.DataFrame, cost_df, cost_years: list,
                       fallback: dict) -> pd.DataFrame:
    yearly = {}
    if cost_df is not None:
        for yr in cost_years:
            col = f"Cost_{yr}"
            if col in cost_df.columns:
                yearly[yr] = cost_df.set_index("SKU")[col].to_dict()

    df["_unit_cost"] = np.nan

    for yr in sorted(df["Year"].dropna().unique().astype(int)):
        mask   = df["Year"] == yr
        lookup = yearly.get(yr, {})
        df.loc[mask, "_unit_cost"] = df.loc[mask, "SKU"].map(lookup)

        missing = mask & df["_unit_cost"].isna()
        if missing.any() and fallback:
            df.loc[missing, "_unit_cost"] = df.loc[missing, "SKU"].map(fallback)

    v = df["_unit_cost"].notna()
    df["Total_Cost"]      = np.nan
    df["Total_Profit"]    = np.nan
    df["Profit_Margin_%"] = np.nan

    df.loc[v, "Total_Cost"]      = (df.loc[v, "_unit_cost"] * df.loc[v, "bal Qty"]).astype("float32")
    df.loc[v, "Total_Profit"]    = (df.loc[v, "bal Value"]  - df.loc[v, "Total_Cost"]).astype("float32")
    df.loc[v, "Profit_Margin_%"] = (df.loc[v, "Total_Profit"] / df.loc[v, "bal Value"] * 100).astype("float32")

    df.drop(columns=["_unit_cost"], inplace=True)

    valid_margin = df.loc[v & df["Profit_Margin_%"].notna() & np.isfinite(df["Profit_Margin_%"]), "Profit_Margin_%"]
    print(f"  profit rows : {v.sum():,}")
    print(f"  total profit: {df['Total_Profit'].sum():,.0f} SAR")
    print(f"  avg margin  : {valid_margin.mean():.1f}%")
    return df


# ── Stage 3 · Enrichment loaders ─────────────────────────────────────────────

def _load_stock(path: Path):
    if not path.exists():
        return None
    df = pd.read_excel(path)
    rename = {}
    for col in df.columns:
        cl = col.lower()
        if "sku" in cl:           rename[col] = "SKU"
        elif "stock" in cl:       rename[col] = "CURRENT_STOCK"
        elif "outstanding" in cl: rename[col] = "OUTSTANDING"
        elif cl.strip() == "pr":  rename[col] = "PR"

    if "SKU" not in rename.values():
        return None

    df = df.rename(columns=rename)
    df["SKU"] = _clean_sku(df["SKU"])
    for col in ["CURRENT_STOCK", "OUTSTANDING", "PR"]:
        if col not in df.columns:
            df[col] = 0
        df[col] = df[col].fillna(0).astype("int32")
    df["EFFECTIVE_STOCK"] = (df["CURRENT_STOCK"] + df["OUTSTANDING"] + df["PR"]).astype("int32")
    return df[["SKU", "CURRENT_STOCK", "OUTSTANDING", "PR", "EFFECTIVE_STOCK"]]


def _load_outlets(path: Path):
    if not path.exists():
        return None
    df = pd.read_excel(path)
    aliases = {
        "Outlet_Names"        : "Outlet Name",
        "Outlets"             : "Outlet_Type",
        "area_master_category": "Area_Master_Category",
        "country"             : "Country",
    }
    rename = {}
    for src, tgt in aliases.items():
        match = next((c for c in df.columns if c.strip().lower() == src.lower()), None)
        if match:
            rename[match] = tgt

    if "Outlet Name" not in rename.values():
        return None

    df = df[list(rename)].rename(columns=rename)
    for col in df.select_dtypes("object").columns:
        df[col] = df[col].astype(str).str.strip()
    df = df[df["Outlet Name"].notna() & ~df["Outlet Name"].isin(["", "nan"])]
    return _dedup(df, "Outlet Name")


def _load_catalogs(path: Path):
    if not path.exists():
        return None
    df = pd.read_excel(path)
    col_map = {}
    for col in df.columns:
        cl = col.strip().lower()
        if "catalog" in cl and "no" in cl:   col_map[col] = "CATALOG_NO"
        elif "collection" in cl:             col_map[col] = "COLLECTION_NAME"
        elif "serial" in cl:                 col_map[col] = "SERIAL"
        elif "sku" in cl or "artcode" in cl: col_map[col] = "SKU"

    if "SKU" not in col_map.values():
        return None

    df = df.rename(columns=col_map)
    df["SKU"] = _clean_sku(df["SKU"])
    str_cols = [c for c in ["CATALOG_NO", "COLLECTION_NAME", "SERIAL"] if c in df.columns]
    for col in str_cols:
        df[col] = df[col].astype(str).str.strip().replace("nan", "")

    def _join_unique(x):
        return ", ".join(sorted({str(v).strip() for v in x if pd.notna(v) and str(v).strip() not in ("", "nan")}))

    keep = ["SKU"] + str_cols
    df   = df[keep].groupby("SKU", as_index=False).agg({c: _join_unique for c in str_cols})
    for col in str_cols:
        df[col] = df[col].replace({"": pd.NA})
    return df


def _load_artprint_texture(path: Path):
    if not path.exists():
        return None
    df = pd.read_excel(path)
    sku_col = _find_col(df, ["sku"])
    if not sku_col:
        return None

    rename = {sku_col: "SKU"}
    for col in df.columns:
        cl = col.lower()
        if "artpatern" in cl or "artpattern" in cl: rename[col] = "ARTPATERN"
        elif "texture" in cl:                        rename[col] = "TEXTURE"
        elif "sub category" in cl:                   rename[col] = "SUB_CATEGORY"

    df = df.rename(columns=rename)
    df["SKU"] = _clean_sku(df["SKU"])

    keep = ["SKU"]
    for col in ["ARTPATERN", "TEXTURE", "SUB_CATEGORY"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            keep.append(col)

    if "ARTPATERN" in df.columns:
        df["_sec"] = df["SKU"].str[:4]
        plain_map = (
            df.groupby("_sec")["ARTPATERN"]
            .apply(lambda x: "YES" if set(x.str.upper().unique()) == {"PLAIN"} else "NO")
            .rename("IS_PLAIN_SECTION")
            .reset_index()
            .rename(columns={"_sec": "_sec_key"})
        )
        df = df.merge(plain_map, left_on="_sec", right_on="_sec_key", how="left")
        df.drop(columns=["_sec", "_sec_key"], inplace=True)
        keep.append("IS_PLAIN_SECTION")

    return df[keep]


def _load_first_inv(path: Path):
    if not path.exists():
        return None
    df = pd.read_excel(path)
    sc = _find_col(df, ["section"])
    dc = _find_col(df, ["date", "تاريخ", "year", "سنة", "inv"])
    if not sc or not dc:
        return None
    df = df.rename(columns={sc: "Section", dc: "First_Inv_Date"})
    df["Section"] = _clean_sku(df["Section"])
    raw    = df["First_Inv_Date"].astype(str).str.strip()
    parsed = pd.to_datetime(raw, format="%m/%d/%Y", errors="coerce")
    nat    = parsed.isna()
    if nat.any():
        parsed[nat] = pd.to_datetime(raw[nat], dayfirst=False, errors="coerce")
    df["First_Inv_Date"] = parsed
    return df[["Section", "First_Inv_Date"]]


def _load_mahmoud_class(path: Path):
    if not path.exists():
        return None
    df = pd.read_excel(path)
    sc = _find_col(df, ["sku"])
    if sc and "SKU" not in df.columns:
        df = df.rename(columns={sc: "SKU"})
    if "SKU" not in df.columns:
        return None
    df["SKU"] = _clean_sku(df["SKU"])
    return df


def _simple_sku_file(path: Path, val_keywords: list, target_col: str,
                     dtype: str = "str"):
    if not path.exists():
        return None
    df = pd.read_excel(path)
    sc = _find_col(df, ["sku"])
    vc = _find_col(df, val_keywords)
    if not sc or not vc:
        return None
    df = df.rename(columns={sc: "SKU", vc: target_col})
    df["SKU"] = _clean_sku(df["SKU"])
    if dtype == "float":
        df[target_col] = pd.to_numeric(df[target_col], errors="coerce").astype("float32")
    elif dtype == "date":
        df[target_col] = pd.to_datetime(df[target_col], errors="coerce")
    return df[["SKU", target_col]]


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run() -> pd.DataFrame:
    print("\n── Stage 1 · Sales Ingestion ─────────────────────────────────")
    df = _load_sales(SALES_DIR, EXCLUDED_OUTLETS)

    print("\n── Stage 2 · Cost Assignment & Profit ────────────────────────")
    cost_df, cost_years = _load_yearly_costs(FILES["yearly_costs"])
    fallback            = _load_fallback_costs(FILES["fallback_cost"])
    df = _calculate_profits(df, cost_df, cost_years, fallback)
    del cost_df
    gc.collect()

    print("\n── Stage 3 · Enrichment Loading ──────────────────────────────")
    enrichments = {
        "stock"          : _load_stock(FILES["stock"]),
        "category"       : _simple_sku_file(FILES["category"],        ["category", "فئة"],       "CATEGORY"),
        "supplier"       : _simple_sku_file(FILES["supplier"],        ["supplier", "مورد"],      "Supplier"),
        "entry_date"     : _simple_sku_file(FILES["entry_date"],      ["date", "تاريخ"],         "LAST_ENTRY_DATE",           dtype="date"),
        "nb_days"        : _simple_sku_file(FILES["nb_days"],         ["nb", "days", "avail"],   "Nb. Days (Avail. Balance)", dtype="float"),
        "cost_per_dollar": _simple_sku_file(FILES["cost_per_dollar"], ["cost", "dollar"],        "COST_PER_DOLLAR",           dtype="float"),
        "color"          : _simple_sku_file(FILES["color"],           ["color", "colour", "لون"],"COLOR_NAME"),
        "historical_qty" : _simple_sku_file(FILES["historical_qty"],  ["qty", "old"],            "OLD_QTY",                   dtype="float"),
        "artprint"       : _load_artprint_texture(FILES["artprint_texture"]),
        "catalogs"       : _load_catalogs(FILES["catalogs"]),
        "outlets"        : _load_outlets(FILES["outlets"]),
        "first_inv"      : _load_first_inv(FILES["first_inv"]),
        "mahmoud_class"  : _load_mahmoud_class(FILES["mahmoud_class"]),
    }

    print("\n── Stage 4 · Merge ───────────────────────────────────────────")
    df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)

    for key in ["stock", "category", "supplier", "entry_date", "nb_days",
                "cost_per_dollar", "color", "artprint", "catalogs", "mahmoud_class"]:
        if enrichments.get(key) is not None:
            df = _left_merge(df, enrichments[key], "SKU", key)
            del enrichments[key]
            gc.collect()

    if enrichments.get("historical_qty") is not None:
        n  = len(df)
        df = df.merge(enrichments["historical_qty"], on="SKU", how="outer")
        print(f"  [historical_qty] outer join → {len(df):,} rows (+{len(df) - n:,})")
        del enrichments["historical_qty"]
        gc.collect()

    if enrichments.get("first_inv") is not None and "Section" in df.columns:
        df["Section"] = _clean_sku(df["Section"])
        df = _left_merge(df, enrichments["first_inv"], "Section", "first_inv")
        del enrichments["first_inv"]
        gc.collect()

    if enrichments.get("outlets") is not None:
        df["Outlet Name"] = df["Outlet Name"].astype(str).str.strip()
        df = _left_merge(df, enrichments["outlets"], "Outlet Name", "outlets")
        del enrichments["outlets"]
        gc.collect()

    print("\n── Stage 5 · Cleaning ────────────────────────────────────────")
    df["Sales_Type"] = np.where(df["bal Qty"] >= 100, "CONTRACT", "FAMILY")
    df["Year"]       = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype("string")

    print(f"  shape  : {df.shape}")
    print(f"  memory : {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

    print("\n── Stage 6 · Save ────────────────────────────────────────────")
    out = OUTPUT_DIR / "combined_df.parquet"
    df.to_parquet(out, engine="pyarrow", compression="snappy", index=False)
    print(f"  {out}  ({out.stat().st_size / 1024**2:.1f} MB)")

    return df


if __name__ == "__main__":
    combined_df = run()


