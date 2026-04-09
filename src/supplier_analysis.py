"""
supplier_analysis.py  ·  src/
===============================
Supplier-level analytics: revenue, margin, stock exposure,
activity classification, and YoY comparison.

Usage
-----
    from src.supplier_analysis import run_supplier_analysis
    result = run_supplier_analysis(combined_df, "SUPPLIER NAME")
"""

import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _norm_sku(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(r"\.0+$", "", regex=True).str.strip().str.upper()


def _find_col(df: pd.DataFrame, keywords: list) -> str | None:
    for col in df.columns:
        if any(k in col.lower() for k in keywords):
            return col
    return None


# ── Core analysis ─────────────────────────────────────────────────────────────

def _build_sku_table(supplier_df: pd.DataFrame,
                     forecast_df: pd.DataFrame | None,
                     adjusted_cost_path: str | None) -> pd.DataFrame:
    """
    Build one row per SKU with all metrics.

    Steps
    -----
    1. Optionally load an adjusted cost file and override unit cost per SKU
    2. Recompute Total_Cost, Total_Profit, Margin_% with the final cost
    3. Aggregate YoY sales (qty + value) into pivot columns
    4. Split current-year sales by CONTRACT vs FAMILY
    5. Compute stock exposure, turnover ratio, activity classification
    6. Merge forecast reorder quantities if provided
    """
    df = supplier_df.copy()
    today = pd.Timestamp.now()
    current_year = today.year

    # ── Step 1 · Optional adjusted cost ──────────────────────────────────
    if adjusted_cost_path and Path(adjusted_cost_path).exists():
        adj = pd.read_excel(adjusted_cost_path)
        sc  = _find_col(adj, ["sku"])
        cc  = _find_col(adj, ["cost", "سعر"])
        if sc and cc:
            adj = adj[[sc, cc]].rename(columns={sc: "SKU", cc: "Adjusted_Cost"})
            adj["SKU"] = _norm_sku(adj["SKU"])
            df["SKU"]  = _norm_sku(df["SKU"])
            df = df.merge(adj, on="SKU", how="left")
            df["Final_Cost"] = df["Adjusted_Cost"].fillna(df.get("Cost", 0))
            print(f"  adjusted costs applied to {df['Adjusted_Cost'].notna().sum():,} rows")
        else:
            df["Final_Cost"] = df.get("Cost", 0)
    else:
        df["Final_Cost"] = df.get("Cost", 0)

    # ── Step 2 · Financials ───────────────────────────────────────────────
    df["Total_Cost"]   = df["Final_Cost"] * df["bal Qty"]
    df["Total_Profit"] = df["bal Value"]  - df["Total_Cost"]
    df["Margin_%"]     = np.where(df["bal Value"] > 0,
                                   df["Total_Profit"] / df["bal Value"] * 100, 0)

    df["Year"]  = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month

    # ── Step 3 · YoY pivot ────────────────────────────────────────────────
    yearly = df.groupby(["SKU", "Year"]).agg(
        Qty   = ("bal Qty",      "sum"),
        Sales = ("bal Value",    "sum"),
        Profit= ("Total_Profit", "sum"),
    ).reset_index()

    qty_piv = (yearly.pivot(index="SKU", columns="Year", values="Qty")
               .fillna(0).add_prefix("Qty_"))
    val_piv = (yearly.pivot(index="SKU", columns="Year", values="Sales")
               .fillna(0).add_prefix("Sales_"))
    pro_piv = (yearly.pivot(index="SKU", columns="Year", values="Profit")
               .fillna(0).add_prefix("Profit_"))

    qty_piv.columns = [f"Qty_{int(c.split('_')[1])}"    for c in qty_piv.columns]
    val_piv.columns = [f"Sales_{int(c.split('_')[1])}"  for c in val_piv.columns]
    pro_piv.columns = [f"Profit_{int(c.split('_')[1])}" for c in pro_piv.columns]

    # ── Step 4 · Current-year CONTRACT / FAMILY split ─────────────────────
    cy_df = df[df["Year"] == current_year].copy()
    if "Sales_Type" in cy_df.columns and len(cy_df):
        type_piv = (
            cy_df.groupby(["SKU", "Sales_Type"])["bal Qty"].sum()
            .unstack("Sales_Type", fill_value=0)
            .add_prefix(f"Qty_{current_year}_")
            .reset_index()
        )
    else:
        type_piv = None

    # ── Step 5 · Static SKU attributes (latest record per SKU) ───────────
    static_cols = ["SKU", "Section", "CATEGORY", "Supplier", "Final_Cost"]
    stock_cols  = [c for c in ["CURRENT_STOCK", "OUTSTANDING", "PR",
                                "EFFECTIVE_STOCK", "LAST_ENTRY_DATE",
                                "First_Inv_Date", "ARTPATERN", "COLOR_NAME",
                                "IS_PLAIN_SECTION"] if c in df.columns]

    latest = df.sort_values("Date").groupby("SKU").last().reset_index()
    sku_base = latest[static_cols + stock_cols].copy()

    if "EFFECTIVE_STOCK" not in sku_base.columns:
        sku_base["EFFECTIVE_STOCK"] = (
            sku_base.get("CURRENT_STOCK", pd.Series(0, index=sku_base.index)).fillna(0) +
            sku_base.get("OUTSTANDING",   pd.Series(0, index=sku_base.index)).fillna(0) +
            sku_base.get("PR",            pd.Series(0, index=sku_base.index)).fillna(0)
        )

    last_sale = df.groupby("SKU")["Date"].max().reset_index().rename(columns={"Date": "Last_Sale_Date"})
    sku_base  = sku_base.merge(last_sale, on="SKU", how="left")

    # ── Merge pivots ──────────────────────────────────────────────────────
    for frame in [qty_piv.reset_index(), val_piv.reset_index(),
                  pro_piv.reset_index()]:
        sku_base = sku_base.merge(frame, on="SKU", how="left")

    if type_piv is not None:
        sku_base = sku_base.merge(type_piv, on="SKU", how="left")

    # ── Step 5 · Derived metrics ──────────────────────────────────────────
    sku_base["Days_Since_Last_Sale"] = (today - sku_base["Last_Sale_Date"]).dt.days
    sku_base["Stock_Value"]          = sku_base["CURRENT_STOCK"].fillna(0) * sku_base["Final_Cost"].fillna(0)

    cy_sales_col = f"Sales_{current_year}"
    if cy_sales_col in sku_base.columns:
        sku_base["Stock_Turnover"] = np.where(
            sku_base["Stock_Value"] > 0,
            sku_base[cy_sales_col] / sku_base["Stock_Value"], 0)
    else:
        sku_base["Stock_Turnover"] = 0

    # YoY growth (current vs prior year)
    prior_col = f"Sales_{current_year - 1}"
    if cy_sales_col in sku_base.columns and prior_col in sku_base.columns:
        sku_base["YoY_Growth_%"] = np.where(
            sku_base[prior_col] > 0,
            (sku_base[cy_sales_col] - sku_base[prior_col]) / sku_base[prior_col] * 100, 0)
    else:
        sku_base["YoY_Growth_%"] = 0

    # Activity classification
    cy_qty_col = f"Qty_{current_year}"
    if cy_qty_col in sku_base.columns:
        q75 = sku_base[cy_qty_col].quantile(0.75)
        q25 = sku_base[cy_qty_col].quantile(0.25)
        sku_base["Activity"] = np.select(
            [
                (sku_base[cy_qty_col] > q75)  & (sku_base["Days_Since_Last_Sale"] < 30),
                (sku_base[cy_qty_col] > 0)     & (sku_base[cy_qty_col] <= q75),
                (sku_base[cy_qty_col] > 0)     & (sku_base[cy_qty_col] < q25),
                sku_base[cy_qty_col] == 0,
            ],
            ["Very Active", "Active", "Slow Moving", "Inactive"],
            default="Active",
        )
    else:
        sku_base["Activity"] = "Unknown"

    # ── Step 6 · Merge forecast ───────────────────────────────────────────
    if forecast_df is not None:
        fc = forecast_df.copy()
        fc["SKU"] = _norm_sku(fc["SKU"])
        fc_cols   = ["SKU"] + [c for c in ["Reorder_Qty", "Monthly_Demand_Final",
                                             "Days_Coverage", "STATUS",
                                             "Target_Stock_7M"] if c in fc.columns]
        sku_base["SKU"] = _norm_sku(sku_base["SKU"])
        sku_base = sku_base.merge(fc[fc_cols], on="SKU", how="left")

    sku_base = sku_base.fillna(0)
    num_cols = sku_base.select_dtypes(include=[np.number]).columns
    sku_base[num_cols] = sku_base[num_cols].round(2)

    return sku_base


# ── Collection-level table ────────────────────────────────────────────────────

def _build_collection_table(supplier_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate key metrics at Collection (Section) level."""
    current_year = pd.Timestamp.now().year
    df = supplier_df.copy()
    df["Section"] = df["SKU"].astype(str).str[:4]

    coll = df.groupby("Section").agg(
        SKU_Count   = ("SKU",         "nunique"),
        Total_Sales = ("bal Value",   "sum"),
        Total_Qty   = ("bal Qty",     "sum"),
        Total_Profit= ("Total_Profit","sum") if "Total_Profit" in df.columns else ("bal Value", lambda x: 0),
    ).reset_index()

    coll["Margin_%"] = np.where(
        coll["Total_Sales"] > 0,
        coll["Total_Profit"] / coll["Total_Sales"] * 100, 0)

    cy = df[df["Date"].dt.year == current_year].groupby("Section")["bal Value"].sum().rename(f"Sales_{current_year}")
    py = df[df["Date"].dt.year == current_year - 1].groupby("Section")["bal Value"].sum().rename(f"Sales_{current_year-1}")
    coll = coll.merge(cy, on="Section", how="left").merge(py, on="Section", how="left")

    coll["YoY_Growth_%"] = np.where(
        coll.get(f"Sales_{current_year-1}", pd.Series(0, index=coll.index)).fillna(0) > 0,
        (coll.get(f"Sales_{current_year}", pd.Series(0, index=coll.index)).fillna(0) -
         coll.get(f"Sales_{current_year-1}", pd.Series(0, index=coll.index)).fillna(0)) /
        coll.get(f"Sales_{current_year-1}", pd.Series(1, index=coll.index)).fillna(1) * 100, 0)

    return coll.sort_values("Total_Sales", ascending=False)


# ── Excel export ──────────────────────────────────────────────────────────────

def _export(sku_df: pd.DataFrame, coll_df: pd.DataFrame,
            supplier_name: str, output_dir: Path) -> Path:
    safe  = "".join(c for c in supplier_name if c.isalnum() or c in " _-")[:40]
    ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
    path  = output_dir / f"Supplier_{safe}_{ts}.xlsx"
    current_year = pd.Timestamp.now().year

    cy_sales = sku_df.get(f"Sales_{current_year}", pd.Series(0)).sum()
    cy_qty   = sku_df.get(f"Qty_{current_year}",   pd.Series(0)).sum()

    summary = pd.DataFrame([
        {"Metric": "Supplier",             "Value": supplier_name},
        {"Metric": "Analysis Date",        "Value": datetime.now().strftime("%Y-%m-%d")},
        {"Metric": "Total SKUs",           "Value": f"{len(sku_df):,}"},
        {"Metric": "Total Collections",   "Value": f"{sku_df['Section'].nunique():,}"},
        {"Metric": "",                     "Value": ""},
        {"Metric": f"Sales {current_year}","Value": f"{cy_sales:,.0f} SAR"},
        {"Metric": f"Qty {current_year}",  "Value": f"{cy_qty:,.0f}"},
        {"Metric": "Total Stock Value",    "Value": f"{sku_df['Stock_Value'].sum():,.0f} SAR"},
        {"Metric": "",                     "Value": ""},
        {"Metric": "Very Active SKUs",     "Value": (sku_df["Activity"] == "Very Active").sum()},
        {"Metric": "Active SKUs",          "Value": (sku_df["Activity"] == "Active").sum()},
        {"Metric": "Slow Moving SKUs",     "Value": (sku_df["Activity"] == "Slow Moving").sum()},
        {"Metric": "Inactive SKUs",        "Value": (sku_df["Activity"] == "Inactive").sum()},
    ])

    with pd.ExcelWriter(path, engine="openpyxl") as w:
        summary.to_excel(w,  sheet_name="Summary",          index=False)
        sku_df.to_excel(w,   sheet_name="SKU Analysis",     index=False)
        coll_df.to_excel(w,  sheet_name="Collections",      index=False)

        reorder = sku_df[sku_df.get("Reorder_Qty", pd.Series(0, index=sku_df.index)).fillna(0) > 0]
        if len(reorder):
            reorder.sort_values("Reorder_Qty", ascending=False).to_excel(
                w, sheet_name="Reorder Required", index=False)

        slow = sku_df[sku_df["Activity"].isin(["Slow Moving", "Inactive"])]
        if len(slow):
            slow.sort_values("Stock_Value", ascending=False).to_excel(
                w, sheet_name="Slow & Inactive", index=False)

    return path


# ── Public API ────────────────────────────────────────────────────────────────

def run_supplier_analysis(
    combined_df: pd.DataFrame,
    supplier_name: str,
    forecast_df: pd.DataFrame | None = None,
    adjusted_cost_path: str | None = None,
    output_dir: str = ".",
) -> dict:
    """
    Full supplier analysis pipeline.

    Parameters
    ----------
    combined_df         : Main sales DataFrame.
    supplier_name       : Exact supplier name string as it appears in the data.
    forecast_df         : Output of run_demand_forecast() — optional.
                          When provided, reorder quantities are merged onto the
                          SKU table so the supplier report includes order actions.
    adjusted_cost_path  : Path to an Excel file with revised unit costs.
                          Columns: SKU, Cost (or سعر).
                          Used to override the cost in combined_df for this
                          supplier if costs have been renegotiated.
    output_dir          : Directory for Excel output.

    Returns
    -------
    dict with keys:
        sku_df        : SKU-level analysis DataFrame
        collection_df : Collection-level summary DataFrame
        supplier_df   : Filtered raw transactions for this supplier
        excel_path    : Path to saved Excel file
    """
    print(f"\n── Supplier Analysis  ·  {supplier_name} ──────────────────────")

    df = combined_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["SKU"]  = df["SKU"].astype(str).str.strip()

    supplier_df = df[df["Supplier"] == supplier_name].copy()
    if len(supplier_df) == 0:
        raise ValueError(f"No data found for supplier: '{supplier_name}'")

    print(f"  records    : {len(supplier_df):,}")
    print(f"  SKUs       : {supplier_df['SKU'].nunique():,}")
    print(f"  date range : {supplier_df['Date'].min().date()} → {supplier_df['Date'].max().date()}")

    print("\n  Building SKU table")
    sku_df = _build_sku_table(supplier_df, forecast_df, adjusted_cost_path)

    print("  Building collection table")
    coll_df = _build_collection_table(supplier_df)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    print("\n  Saving Excel")
    path = _export(sku_df, coll_df, supplier_name, out)
    print(f"  saved  ·  {path.name}")

    current_year = pd.Timestamp.now().year
    print(f"\n── Summary ───────────────────────────────────────────────────")
    print(f"  SKUs           : {len(sku_df):,}")
    print(f"  Collections    : {sku_df['Section'].nunique():,}")
    print(f"  Very Active    : {(sku_df['Activity'] == 'Very Active').sum():,}")
    print(f"  Slow/Inactive  : {sku_df['Activity'].isin(['Slow Moving','Inactive']).sum():,}")
    if f"Sales_{current_year}" in sku_df.columns:
        print(f"  Sales {current_year}    : {sku_df[f'Sales_{current_year}'].sum():,.0f} SAR")
    print(f"  Stock Value    : {sku_df['Stock_Value'].sum():,.0f} SAR")

    return {
        "sku_df"       : sku_df,
        "collection_df": coll_df,
        "supplier_df"  : supplier_df,
        "excel_path"   : path,
    }
