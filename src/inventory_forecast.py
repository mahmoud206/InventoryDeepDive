"""
inventory_forecast.py  ·  src/
================================
WMA-based demand forecasting and reorder calculation.

Method
------
    1. Filter to FAMILY sales only (excludes CONTRACT bulk orders)
    2. Aggregate daily sales by SKU over the last 365 days
    3. Split into four rolling 90-day quarters (Q1 = most recent)
    4. Compute per-quarter average daily sales (ADS) using a blended
       denominator: mix of calendar days and actual selling days
       → prevents demand inflation for intermittent SKUs
    5. Assign weights dynamically if SKU has ≥ 30 active days,
       otherwise fall back to standard weights (40/30/20/10)
    6. WMA = Σ(ADS_q × W_q) across four quarters
    7. Adjust upward for SKUs with recent stockout gaps
    8. Compute reorder qty = Target(7M) − Effective_Stock, clipped at 0

Usage
-----
    from src.inventory_forecast import run_demand_forecast
    forecast_df, path = run_demand_forecast(combined_df, nb_days_df)
"""

import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ── Configuration ─────────────────────────────────────────────────────────────

LOOKBACK_DAYS          = 365   # historical window
FORWARD_COVERAGE_MONTHS= 7     # target stock coverage
STOCKOUT_THRESHOLD     = 10    # units; at or below = at-risk
STOCKOUT_CAP           = 1.5   # max upward adjustment factor
MIN_DATA_COVERAGE      = 0.50  # fraction of lookback period required
MIN_ACTIVE_DAYS        = 30    # active selling days to enable dynamic weights

# Quarterly blending
QUARTER_DAYS           = 90
BLEND_FACTOR           = 0.4   # 0 = pure calendar days | 1 = pure selling days
MIN_SALES_DAYS_BLEND   = 10    # below this, use calendar days only
MAX_DAILY_MULTIPLIER   = 1.2   # cap per-quarter ADS vs yearly average


# ── Stage 1 · Quarterly metrics ───────────────────────────────────────────────

def _quarterly_metrics(daily: pd.DataFrame, today: pd.Timestamp) -> pd.DataFrame:
    """
    Compute per-SKU per-quarter sales sum, selling-day count, and blended ADS.

    Blended denominator:
        adjusted_days = α × actual_selling_days + (1−α) × 90

    This avoids inflating ADS for SKUs that only sold on a few days in a quarter
    while still giving credit for concentrated selling patterns.
    """
    cutoffs = {
        "Q1": today - pd.Timedelta(days=90),
        "Q2": today - pd.Timedelta(days=180),
        "Q3": today - pd.Timedelta(days=270),
    }

    d = daily.copy()
    d["Quarter"] = pd.cut(
        d["Date"],
        bins=[pd.Timestamp.min, cutoffs["Q3"], cutoffs["Q2"], cutoffs["Q1"], today],
        labels=["Q4", "Q3", "Q2", "Q1"],
        include_lowest=True,
    )

    # Yearly anchor — prevents quarterly caps being mis-applied for new SKUs
    yearly = (
        d[d["bal Qty"] > 0]
        .groupby("SKU")["bal Qty"]
        .agg(Total_1Y="sum", Active_Days_1Y="count")
        .reset_index()
    )
    yearly["Yearly_Avg_Daily"] = np.where(
        yearly["Active_Days_1Y"] > 0,
        yearly["Total_1Y"] / yearly["Active_Days_1Y"], 0)

    # Quarterly aggregation — selling days only
    qdf = (
        d[d["bal Qty"] > 0]
        .groupby(["SKU", "Quarter"])
        .agg(Sum=("bal Qty", "sum"), Days=("bal Qty", "count"))
        .reset_index()
    )

    wide = (
        qdf.pivot(index="SKU", columns="Quarter", values=["Sum", "Days"])
        .fillna(0)
    )
    wide.columns = [f"{q}_{m}" for m, q in wide.columns]
    wide = wide.reset_index().merge(yearly[["SKU", "Yearly_Avg_Daily"]], on="SKU", how="left")

    for q in ["Q1", "Q2", "Q3", "Q4"]:
        days_col = f"{q}_Days"
        sum_col  = f"{q}_Sum"
        avg_col  = f"{q}_Avg_Daily"

        if days_col not in wide.columns: wide[days_col] = 0
        if sum_col  not in wide.columns: wide[sum_col]  = 0

        adj_days = np.where(
            wide[days_col] >= MIN_SALES_DAYS_BLEND,
            BLEND_FACTOR * wide[days_col] + (1 - BLEND_FACTOR) * QUARTER_DAYS,
            QUARTER_DAYS,
        )
        raw_avg = np.where(wide[days_col] > 0, wide[sum_col] / adj_days, 0)
        cap     = wide["Yearly_Avg_Daily"].fillna(0) * MAX_DAILY_MULTIPLIER
        wide[avg_col] = np.minimum(raw_avg, cap)

    wide.drop(columns=["Yearly_Avg_Daily"], inplace=True)
    return wide


# ── Stage 2 · Dynamic weights ─────────────────────────────────────────────────

def _dynamic_weights(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign per-SKU quarterly weights.

    Active SKUs (≥ MIN_ACTIVE_DAYS selling days total):
        W_q = proportion of total sales in that quarter
        Floor: Q1 ≥ 0.30, others ≥ 0.10  (recency bias)
        Normalised to sum to 1.0

    Sparse SKUs:
        Fixed weights: Q1=0.40, Q2=0.30, Q3=0.20, Q4=0.10
    """
    d = df.copy()
    d["_total_sales"]= d[["Q1_Sum","Q2_Sum","Q3_Sum","Q4_Sum"]].sum(axis=1)
    d["_total_days"] = d[["Q1_Days","Q2_Days","Q3_Days","Q4_Days"]].sum(axis=1)
    active           = d["_total_days"] >= MIN_ACTIVE_DAYS

    floors = {"Q1": 0.30, "Q2": 0.10, "Q3": 0.10, "Q4": 0.10}
    for q, floor in floors.items():
        raw = np.where(d["_total_sales"] > 0, d[f"{q}_Sum"] / d["_total_sales"], 0)
        d[f"_w_{q}"] = np.maximum(raw, floor)

    wsum = d[["_w_Q1","_w_Q2","_w_Q3","_w_Q4"]].sum(axis=1)
    for q in ["Q1","Q2","Q3","Q4"]:
        d[f"_w_{q}"] /= wsum

    std = {"Q1": 0.40, "Q2": 0.30, "Q3": 0.20, "Q4": 0.10}
    for q, w in std.items():
        d[f"W_{q}"] = np.where(active, d[f"_w_{q}"], w)

    d["Weight_Method"] = np.where(active, "Dynamic", "Standard")

    drop = [c for c in d.columns if c.startswith("_")]
    d.drop(columns=drop, inplace=True)
    return d


# ── Stage 3 · WMA + stockout adjustment ──────────────────────────────────────

def _wma(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Weighted Moving Average daily demand and monthly projection."""
    df["ADS_WMA"] = (
        df["Q1_Avg_Daily"] * df["W_Q1"] +
        df["Q2_Avg_Daily"] * df["W_Q2"] +
        df["Q3_Avg_Daily"] * df["W_Q3"] +
        df["Q4_Avg_Daily"] * df["W_Q4"]
    )
    df["Monthly_WMA"]    = df["ADS_WMA"] * 30
    df["Q1_Monthly"]     = df["Q1_Avg_Daily"] * 30
    df["Q2_Monthly"]     = df["Q2_Avg_Daily"] * 30
    df["Q3_Monthly"]     = df["Q3_Avg_Daily"] * 30
    df["Q4_Monthly"]     = df["Q4_Avg_Daily"] * 30
    return df


def _final_demand(df: pd.DataFrame) -> pd.DataFrame:
    """
    Choose between simple average and WMA.

    WMA is used when:
        - Data coverage ≥ 50% of the lookback window  (sufficient history)
        - WMA > simple average                         (WMA only upgrades, never downgrades)

    Otherwise the simple 12-month average is used as the conservative baseline.
    """
    df["Monthly_Demand_Actual"] = df["Total_Sales_1Y"] / 12
    df["Monthly_Demand_Final"]  = df["Monthly_Demand_Actual"]
    df["Forecast_Method"]       = "Actual (Conservative)"

    good     = df["Data_Coverage_%"] >= MIN_DATA_COVERAGE * 100
    wma_wins = df["Monthly_WMA"] > df["Monthly_Demand_Actual"]

    df.loc[good & wma_wins, "Monthly_Demand_Final"] = df.loc[good & wma_wins, "Monthly_WMA"]
    df.loc[good & wma_wins, "Forecast_Method"]      = "WMA"
    return df


def _stockout_adjustment(df: pd.DataFrame, nb_days_df: pd.DataFrame | None) -> pd.DataFrame:
    """
    Upward-adjust demand estimate for SKUs that experienced stockout gaps.

    Logic:
        If a SKU had fewer than STOCKOUT_WINDOW available days in the last 120,
        it likely undersold.  The adjustment factor scales demand by:

            factor = (available + 0.5 × missing) / available

        Capped at STOCKOUT_CAP (1.5×) to avoid over-ordering.
        The 0.5 coefficient assumes lost sales were partially recoverable
        (some demand deferred, some permanently lost).
    """
    df["Days_Stockout_Last_120"] = 0
    df["Stockout_Adjusted"]      = "No"
    df["Stockout_Fill_Method"]   = "N/A"

    if nb_days_df is None:
        return df

    nb = nb_days_df.copy()
    nb["SKU"] = nb["SKU"].astype(str).str.strip()
    df = df.merge(nb[["SKU", "Nb. Days (Avail. Balance)"]], on="SKU", how="left")

    WINDOW = 120
    mask = (
        (df["CURRENT_STOCK"] <= STOCKOUT_THRESHOLD) &
        df["Nb. Days (Avail. Balance)"].notna() &
        df["Nb. Days (Avail. Balance)"].between(0.01, WINDOW - 1)
    )

    df["Days_Stockout_Last_120"] = np.where(
        mask, WINDOW - df["Nb. Days (Avail. Balance)"], 0)

    avail  = df.loc[mask, "Nb. Days (Avail. Balance)"]
    missed = df.loc[mask, "Days_Stockout_Last_120"]
    factor = ((avail + 0.5 * missed) / avail).clip(upper=STOCKOUT_CAP)

    df.loc[mask, "Monthly_Demand_Final"] *= factor
    df.loc[mask, "Stockout_Adjusted"]     = "Yes"
    df.loc[mask, "Stockout_Fill_Method"]  = f"Half-fill (cap {STOCKOUT_CAP}x)"

    return df


# ── Stage 4 · Inventory positions ────────────────────────────────────────────

def _inventory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute reorder quantities and inventory status flags.

        ADS_Final        = Monthly_Demand_Final / 30
        Effective_Stock  = CURRENT_STOCK + OUTSTANDING + PR
        Target_Stock_7M  = Monthly_Demand_Final × 7
        Reorder_Qty      = max(0, Target − Effective_Stock)
        Days_Coverage    = Effective_Stock / ADS_Final
    """
    df["ADS_Final"]        = df["Monthly_Demand_Final"] / 30
    df["Effective_Stock"]  = (
        df["CURRENT_STOCK"].fillna(0) +
        df["OUTSTANDING"].fillna(0) +
        df["PR"].fillna(0)
    ).astype(int)
    df["Target_Stock_7M"]  = (df["Monthly_Demand_Final"] * FORWARD_COVERAGE_MONTHS).astype(int)
    df["Reorder_Qty"]      = (df["Target_Stock_7M"] - df["Effective_Stock"]).clip(lower=0).astype(int)
    df["Days_Coverage"]    = np.where(
        df["ADS_Final"] > 0, df["Effective_Stock"] / df["ADS_Final"], 999).round(1)

    conditions = [
        df["CURRENT_STOCK"] == 0,
        df["CURRENT_STOCK"] <= 30,
        df["Effective_Stock"] > df["Target_Stock_7M"],
        df["Reorder_Qty"] > 0,
    ]
    choices = ["OUT_OF_STOCK", "CRITICAL", "EXCESS", "REORDER"]
    df["STATUS"] = np.select(conditions, choices, default="OK")
    return df


# ── Stage 5 · Excel export ────────────────────────────────────────────────────

def _export(df: pd.DataFrame, today: pd.Timestamp, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"Demand_Forecast_{today.strftime('%Y%m%d_%H%M%S')}.xlsx"

    dynamic_n  = (df["Weight_Method"] == "Dynamic").sum()
    standard_n = (df["Weight_Method"] == "Standard").sum()
    reorder_n  = (df["Reorder_Qty"] > 0).sum()

    summary = pd.DataFrame([
        {"Metric": "Analysis Date",           "Value": today.strftime("%Y-%m-%d")},
        {"Metric": "Total SKUs",              "Value": f"{len(df):,}"},
        {"Metric": "Lookback Window",         "Value": f"{LOOKBACK_DAYS} days"},
        {"Metric": "Coverage Target",         "Value": f"{FORWARD_COVERAGE_MONTHS} months"},
        {"Metric": "",                        "Value": ""},
        {"Metric": "Avg Actual Sales Days",   "Value": f"{df['Actual_Sales_Days'].mean():.0f} / {LOOKBACK_DAYS}"},
        {"Metric": "Avg Data Coverage",       "Value": f"{df['Data_Coverage_%'].mean():.1f}%"},
        {"Metric": "",                        "Value": ""},
        {"Metric": "Dynamic Weights",         "Value": f"{dynamic_n:,}"},
        {"Metric": "Standard Weights",        "Value": f"{standard_n:,}"},
        {"Metric": "",                        "Value": ""},
        {"Metric": "Reorder Required",        "Value": f"{reorder_n:,}"},
        {"Metric": "Total Reorder Qty",       "Value": f"{df['Reorder_Qty'].sum():,.0f}"},
        {"Metric": "Critical (stock ≤ 30)",   "Value": f"{(df['STATUS'] == 'CRITICAL').sum():,}"},
        {"Metric": "Out of Stock",            "Value": f"{(df['STATUS'] == 'OUT_OF_STOCK').sum():,}"},
        {"Metric": "Excess Stock",            "Value": f"{(df['STATUS'] == 'EXCESS').sum():,}"},
    ])

    with pd.ExcelWriter(path, engine="openpyxl") as w:
        summary.to_excel(w, sheet_name="Summary", index=False)
        df.to_excel(w, sheet_name="All SKUs", index=False)

        reorder = df[df["Reorder_Qty"] > 0].sort_values("Reorder_Qty", ascending=False)
        if len(reorder):
            reorder.to_excel(w, sheet_name="Reorder Required", index=False)

        critical = df[df["STATUS"].isin(["CRITICAL", "OUT_OF_STOCK"])]
        if len(critical):
            critical.sort_values("Days_Coverage").to_excel(w, sheet_name="Critical", index=False)

        if "Section" in df.columns:
            section_sum = (
                df.groupby("Section").agg(
                    Total_SKUs        = ("SKU",          "count"),
                    Total_Sales_1Y    = ("Total_Sales_1Y","sum"),
                    Total_Reorder_Qty = ("Reorder_Qty",   "sum"),
                    Effective_Stock   = ("Effective_Stock","sum"),
                )
                .reset_index()
                .sort_values("Total_Reorder_Qty", ascending=False)
            )
            section_sum.to_excel(w, sheet_name="Section Summary", index=False)

    return path


# ── Public API ────────────────────────────────────────────────────────────────

def run_demand_forecast(
    combined_df: pd.DataFrame,
    nb_days_df: pd.DataFrame | None = None,
    analysis_date=None,
    output_dir: str = ".",
) -> tuple:
    """
    WMA-based demand forecast and reorder calculation.

    Parameters
    ----------
    combined_df   : Sales DataFrame — requires Date, SKU, bal Qty,
                    CATEGORY, CURRENT_STOCK, OUTSTANDING.
                    Optional: PR, Section, Sales_Type.
    nb_days_df    : Stockout reference — requires SKU,
                    'Nb. Days (Avail. Balance)'.
    analysis_date : Reference date (defaults to today).
    output_dir    : Directory for Excel output.

    Returns
    -------
    forecast_df : DataFrame  — one row per SKU with all metrics
    path        : Path       — saved Excel file
    """
    today = pd.Timestamp(analysis_date) if analysis_date else pd.Timestamp.now()
    out   = Path(output_dir)

    print(f"\n── Demand Forecast Pipeline  ·  {today.date()} ──────────────────")

    # ── Preprocessing ─────────────────────────────────────────────────────
    df = combined_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["SKU"]  = df["SKU"].astype(str).str.strip()

    for col in ["CURRENT_STOCK", "OUTSTANDING", "bal Qty"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df["PR"] = pd.to_numeric(df.get("PR", 0), errors="coerce").fillna(0)

    has_section = "Section" in df.columns

    if "Sales_Type" in df.columns:
        before = len(df)
        df = df[df["Sales_Type"] == "FAMILY"]
        print(f"  FAMILY filter : {len(df):,} / {before:,} records")
    else:
        print("  Sales_Type not found — using all records")

    cutoff = today - pd.Timedelta(days=LOOKBACK_DAYS)
    df = df[df["Date"] >= cutoff]
    print(f"  Date range    : {df['Date'].min().date()} → {df['Date'].max().date()}")

    # ── Daily aggregation ──────────────────────────────────────────────────
    agg = {"bal Qty": "sum", "CATEGORY": "first",
           "CURRENT_STOCK": "first", "OUTSTANDING": "first", "PR": "first"}
    if has_section:
        agg["Section"] = "first"

    daily = df.groupby(["SKU", "Date"], as_index=False).agg(agg)

    # ── Stage 1: scalar metrics ────────────────────────────────────────────
    selling_days = (
        daily[daily["bal Qty"] > 0]
        .groupby("SKU").size()
        .reset_index(name="Actual_Sales_Days")
    )

    base_agg = {"bal Qty": "sum", "CATEGORY": "first",
                "CURRENT_STOCK": "first", "OUTSTANDING": "first", "PR": "first"}
    if has_section:
        base_agg["Section"] = "first"

    sku = daily.groupby("SKU").agg(base_agg).reset_index()
    sku.rename(columns={"bal Qty": "Total_Sales_1Y"}, inplace=True)
    sku = sku.merge(selling_days, on="SKU", how="left")
    sku["Actual_Sales_Days"] = sku["Actual_Sales_Days"].fillna(0).astype(int)
    sku["Data_Coverage_%"]   = (sku["Actual_Sales_Days"] / LOOKBACK_DAYS * 100).round(1)

    print(f"  SKUs          : {len(sku):,}")
    print(f"  Avg coverage  : {sku['Data_Coverage_%'].mean():.1f}%")

    # ── Stage 2: quarterly metrics → weights → WMA ─────────────────────────
    print("\n  Stage 1 · Quarterly metrics")
    qm  = _quarterly_metrics(daily, today)
    sku = sku.merge(qm, on="SKU", how="left")

    for q in ["Q1","Q2","Q3","Q4"]:
        for s in ["Sum","Days","Avg_Daily"]:
            c = f"{q}_{s}"
            if c not in sku.columns: sku[c] = 0
            else: sku[c] = sku[c].fillna(0)

    print("  Stage 2 · Dynamic weights")
    sku = _dynamic_weights(sku)

    dyn = (sku["Weight_Method"] == "Dynamic").sum()
    print(f"    dynamic  : {dyn:,}  |  standard : {len(sku)-dyn:,}")

    print("  Stage 3 · WMA")
    sku = _wma(sku)
    sku = _final_demand(sku)

    wma_used = (sku["Forecast_Method"] == "WMA").sum()
    print(f"    WMA used : {wma_used:,}  |  actual avg : {len(sku)-wma_used:,}")

    print("  Stage 4 · Stockout adjustment")
    sku = _stockout_adjustment(sku, nb_days_df)
    adjusted = (sku["Stockout_Adjusted"] == "Yes").sum()
    print(f"    adjusted : {adjusted:,} SKUs")

    print("  Stage 5 · Inventory positions")
    sku = _inventory(sku)

    reorder_n = (sku["Reorder_Qty"] > 0).sum()
    critical_n= (sku["STATUS"] == "CRITICAL").sum()
    oos_n     = (sku["STATUS"] == "OUT_OF_STOCK").sum()
    excess_n  = (sku["STATUS"] == "EXCESS").sum()

    print(f"\n── Results ───────────────────────────────────────────────────")
    print(f"  reorder required : {reorder_n:>8,}")
    print(f"  total reorder qty: {sku['Reorder_Qty'].sum():>8,.0f}")
    print(f"  critical         : {critical_n:>8,}")
    print(f"  out of stock     : {oos_n:>8,}")
    print(f"  excess stock     : {excess_n:>8,}")

    print("\n  Stage 6 · Saving Excel")
    path = _export(sku, today, out)
    print(f"  saved  ·  {path.name}")

    return sku, path
