"""
pattern_color.py  ·  src/
==========================
Pattern and color analysis engine.

SKU structure assumed:
    Digits 0-3  → Collection  (4 digits)
    Digits 4-6  → Pattern     (3 digits)
    Digits 7-9  → Color       (3 digits)

Usage
-----
    from src.pattern_color import run_pattern_color_analysis
    results = run_pattern_color_analysis(combined_df)
"""

import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ── SKU parsing ───────────────────────────────────────────────────────────────

def _parse_sku(sku: str) -> dict:
    digits = "".join(c for c in str(sku) if c.isdigit())
    collection = digits[:4]
    pattern    = digits[4:7] if len(digits) >= 7  else ""
    color      = digits[7:10] if len(digits) >= 10 else ""
    return {
        "Collection"         : collection,
        "Pattern"            : pattern,
        "Color"              : color,
        "Collection_Pattern" : f"{collection}-{pattern}" if pattern else collection,
    }


def add_sku_components(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse SKU strings and append Collection, Pattern, Color,
    Collection_Pattern columns.  Safe to call multiple times.
    """
    components = pd.DataFrame(df["SKU"].apply(_parse_sku).tolist(), index=df.index)
    for col in components.columns:
        df[col] = components[col]
    return df


# ── Pattern analysis ──────────────────────────────────────────────────────────

def analyze_patterns(df: pd.DataFrame, min_sales: int = 10) -> tuple:
    """
    Aggregate sales and stock by Collection_Pattern.

    Returns
    -------
    pattern_stats : full pattern table sorted by Performance_Score
    best          : top-10 patterns
    worst         : bottom-10 patterns
    sku_level     : SKU-level aggregation used by color analysis
    """
    has_profit = "Total_Profit" in df.columns
    has_stock  = "CURRENT_STOCK" in df.columns
    has_out    = "OUTSTANDING" in df.columns

    agg_spec = {
        "bal Qty"             : "sum",
        "bal Value"           : "sum",
        "Collection_Pattern"  : "first",
        "Collection"          : "first",
        "Pattern"             : "first",
        "Color"               : "first",
    }
    if has_profit: agg_spec["Total_Profit"]   = "sum"
    if has_stock:  agg_spec["CURRENT_STOCK"]  = "first"
    if has_out:    agg_spec["OUTSTANDING"]    = "first"

    sku_level = df.groupby("SKU").agg(agg_spec).reset_index()

    if not has_profit: sku_level["Total_Profit"]  = 0.0
    if not has_stock:  sku_level["CURRENT_STOCK"] = 0
    if not has_out:    sku_level["OUTSTANDING"]   = 0

    pat_agg = sku_level.groupby("Collection_Pattern").agg(
        SKU_Count     = ("SKU",            "count"),
        Sales_Qty     = ("bal Qty",        "sum"),
        Sales_Value   = ("bal Value",      "sum"),
        Total_Profit  = ("Total_Profit",   "sum"),
        Stock         = ("CURRENT_STOCK",  "sum"),
        Outstanding   = ("OUTSTANDING",    "sum"),
    ).reset_index()

    pat_agg["Margin_%"]           = np.where(pat_agg["Sales_Value"] > 0,
                                              pat_agg["Total_Profit"] / pat_agg["Sales_Value"] * 100, 0)
    pat_agg["Avg_Sales_Per_SKU"]  = pat_agg["Sales_Qty"] / pat_agg["SKU_Count"]
    pat_agg["Stock_Sales_Ratio"]  = pat_agg["Stock"] / pat_agg["Sales_Qty"].replace(0, 1)
    # Performance Score: reward revenue, penalise idle stock
    pat_agg["Performance_Score"]  = pat_agg["Sales_Value"] + pat_agg["Total_Profit"] * 0.5 \
                                    - pat_agg["Stock"] * 10

    pat_agg = pat_agg.sort_values("Performance_Score", ascending=False)
    filtered = pat_agg[pat_agg["Sales_Qty"] >= min_sales].copy()

    return pat_agg, filtered.head(10), filtered.tail(10), sku_level


# ── Color analysis ────────────────────────────────────────────────────────────

def analyze_colors(sku_level: pd.DataFrame) -> tuple:
    """
    Aggregate by Color within each Collection_Pattern, and also globally.

    Returns
    -------
    color_by_pattern : performance per (Pattern, Color) pair
    color_overall    : global color performance sorted by Sales_Value
    best_colors      : top-10 colors
    worst_colors     : bottom-10 colors
    """
    cbp = sku_level.groupby(["Collection_Pattern", "Color"]).agg(
        SKU_Count    = ("SKU",           "count"),
        Sales_Qty    = ("bal Qty",       "sum"),
        Sales_Value  = ("bal Value",     "sum"),
        Total_Profit = ("Total_Profit",  "sum"),
        Stock        = ("CURRENT_STOCK", "sum"),
        Outstanding  = ("OUTSTANDING",   "sum"),
    ).reset_index()

    cbp["Margin_%"]         = np.where(cbp["Sales_Value"] > 0,
                                        cbp["Total_Profit"] / cbp["Sales_Value"] * 100, 0)
    cbp["Performance_Score"]= cbp["Sales_Value"] + cbp["Total_Profit"] * 0.5 \
                              - cbp["Stock"] * 10
    cbp = cbp.sort_values("Performance_Score", ascending=False)

    co = sku_level.groupby("Color").agg(
        SKU_Count    = ("SKU",           "count"),
        Sales_Qty    = ("bal Qty",       "sum"),
        Sales_Value  = ("bal Value",     "sum"),
        Total_Profit = ("Total_Profit",  "sum"),
        Stock        = ("CURRENT_STOCK", "sum"),
        Outstanding  = ("OUTSTANDING",   "sum"),
    ).reset_index()

    co["Margin_%"] = np.where(co["Sales_Value"] > 0,
                               co["Total_Profit"] / co["Sales_Value"] * 100, 0)
    co = co.sort_values("Sales_Value", ascending=False)

    return cbp, co, co.head(10), co.tail(10)


# ── Excel export ──────────────────────────────────────────────────────────────

def export_results(pattern_stats, best, worst,
                   color_by_pattern, color_overall,
                   best_colors, worst_colors,
                   output_dir: str = ".") -> str | None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = str(Path(output_dir) / f"PATTERN_COLOR_ANALYSIS_{ts}.xlsx")

    try:
        with pd.ExcelWriter(path, engine="openpyxl") as w:
            pattern_stats.to_excel(w,    sheet_name="All Patterns",     index=False)
            best.to_excel(w,             sheet_name="Best Patterns",     index=False)
            worst.to_excel(w,            sheet_name="Worst Patterns",    index=False)
            color_by_pattern.to_excel(w, sheet_name="Colors by Pattern", index=False)
            color_overall.to_excel(w,    sheet_name="Colors Overall",    index=False)
            best_colors.to_excel(w,      sheet_name="Best Colors",       index=False)
            worst_colors.to_excel(w,     sheet_name="Worst Colors",      index=False)
        print(f"  saved  ·  {path}")
        return path
    except PermissionError:
        print(f"  save failed: file may be open in Excel")
        return None
    except Exception as e:
        print(f"  save failed: {e}")
        return None


# ── Public API ────────────────────────────────────────────────────────────────

def run_pattern_color_analysis(df: pd.DataFrame,
                                min_sales: int = 10,
                                save_excel: bool = True,
                                output_dir: str = ".") -> dict:
    """
    Full pattern and color analysis pipeline.

    Parameters
    ----------
    df          : combined_df — requires SKU, bal Qty, bal Value.
                  Optional: CURRENT_STOCK, OUTSTANDING, Total_Profit.
    min_sales   : minimum qty threshold for best/worst pattern lists.
    save_excel  : write Excel output.
    output_dir  : directory for Excel output.

    Returns
    -------
    dict with keys:
        df_updated, sku_level, patterns, best, worst,
        colors, color_overall, export_file
    """
    print("\n── Pattern & Color Analysis ──────────────────────────────────")

    df_enhanced = add_sku_components(df.copy())

    pattern_stats, best, worst, sku_level = analyze_patterns(df_enhanced, min_sales)
    color_by_pattern, color_overall, best_colors, worst_colors = analyze_colors(sku_level)

    print(f"  collections : {df_enhanced['Collection'].nunique():,}")
    print(f"  patterns    : {df_enhanced['Collection_Pattern'].nunique():,}")
    print(f"  colors      : {df_enhanced['Color'].nunique():,}")

    export_file = None
    if save_excel:
        export_file = export_results(
            pattern_stats, best, worst,
            color_by_pattern, color_overall,
            best_colors, worst_colors,
            output_dir,
        )

    return {
        "df_updated"    : df_enhanced,
        "sku_level"     : sku_level,
        "patterns"      : pattern_stats,
        "best"          : best,
        "worst"         : worst,
        "colors"        : color_by_pattern,
        "color_overall" : color_overall,
        "export_file"   : export_file,
    }
