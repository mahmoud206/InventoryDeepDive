# Inventory Analysis — Methodology

Full reference for every analytical method used across the five notebooks.

---

## Table of Contents

1. [Data Pipeline](#1-data-pipeline)
2. [SKU & Section Classification](#2-sku--section-classification)
3. [Pattern & Color Analysis](#3-pattern--color-analysis)
4. [Demand Forecasting — WMA](#4-demand-forecasting--wma)
5. [Supplier Analysis](#5-supplier-analysis)
6. [Shared Concepts](#6-shared-concepts)
7. [Output Column Reference](#7-output-column-reference)

---

## 1 · Data Pipeline

**Notebook:** `01_data_loading.ipynb` · **Module:** `src/data_loader.py`

### What it does

Loads all yearly CSV sales files, joins 13 auxiliary Excel files (costs, stock, categories, suppliers, outlets, catalogs, colors, patterns), calculates profit, and writes a single Parquet file used by all downstream notebooks.

### Key design decisions

**Outlet exclusion** — Internal and non-commercial outlets (factory catalogues, administrative branches) are excluded by keyword match on `Outlet Name`. Including them would distort per-unit demand and margin calculations because their transactions do not reflect market pricing.

**Yearly cost assignment** — Unit costs are matched by year before profit is calculated. A SKU sold in 2022 uses the 2022 cost sheet, not the current one. This prevents historical margin figures from being distorted by cost changes made after the fact.

**Left joins for all enrichments** — Every enrichment file is joined as a left join onto the sales table. This means: if a SKU has no category record, it still appears in the output with a null category. Rows are never lost due to missing enrichment data. The one exception is `historical_qty` (pre-2019 data), which uses an outer join to preserve SKUs that existed before the main sales period.

**Row-count guard** — Every left join asserts that the row count does not change after the merge. If it does, the join key had duplicates in the enrichment file, which would silently multiply revenue figures. The pipeline raises an explicit error rather than proceeding with corrupted data.

**Parquet output** — Snappy-compressed Parquet is ~3–5× smaller than CSV and loads in seconds for subsequent notebooks. Column types are preserved exactly, including pandas `Int64` (nullable integer) for year values.

---

## 2 · SKU & Section Classification

**Notebook:** `04_sku_collection_classification.ipynb` · **Module:** `src/classification.py`

### Robust statistics

Sales data is right-skewed and log-normal. Standard mean/std statistics are unreliable because a single bulk order can be 50× the median transaction.

**Median Absolute Deviation (MAD):**

```
MAD = median(|xᵢ − median(x)|)
```

**Robust Coefficient of Variation:**

```
Robust CV = (MAD / median) × 1.4826
```

The 1.4826 factor is `1 / Φ⁻¹(0.75)` where Φ is the normal CDF. It makes MAD consistent with standard deviation when the data is Gaussian, while remaining stable when it is not.

| Robust CV | Interpretation |
|---|---|
| < 0.3 | Predictable demand |
| 0.3 – 0.6 | Moderate variability |
| 0.6 – 1.0 | High variability |
| > 1.0 | Extreme — normal for seasonal products |

### Trend detection — multi-signal voting

No single trend test is reliable on noisy retail data. Four independent signals are combined via weighted voting.

**Signal 1 — Holt's double exponential smoothing (numpy EMA):**

```
Level:  Lₜ = α·yₜ + (1−α)(Lₜ₋₁ + Tₜ₋₁)
Trend:  Tₜ = β(Lₜ − Lₜ₋₁) + (1−β)Tₜ₋₁
```

Parameters: α = 0.2 (level), β = 0.1 (trend). Low values give recent data more weight without overreacting to single-period spikes. The mean trend component over the last 3 periods is used as the signal value.

**Signal 2 — Spearman rank correlation:**

```
ρ = 1 − 6Σdᵢ² / (n(n²−1))
```

Spearman is used instead of Mann-Kendall because both detect monotonic trends, but Spearman's scipy implementation runs in O(n log n) — approximately 100× faster per call — which matters when processing 133K SKUs.

**Signal 3 — Period-over-period change:**

```
PoP% = (mean(last 3 months) − mean(prior 3 months)) / mean(prior 3 months) × 100
```

**Signal 4 — Year-over-year change (seasonal SKUs only):**

```
YoY% = (last 3 months − same 3 months last year) / prior year × 100
```

Only activated when the STL seasonal strength exceeds 0.3.

**Weighted voting:**

| Signal | Weight | Threshold to vote |
|---|---|---|
| Holt trend component | 1.5 | > ±200 units |
| Spearman ρ | 2.0 | ρ > ±0.1, p < 0.15 |
| Period-over-period | 1.5 | > ±8% |
| Year-over-year | 2.0 | > ±10% |

Direction: **Growing** if growing-vote share > 60%, **Declining** if declining-vote share > 60%, **Stable** otherwise.

Confidence is adjusted down when directional consistency < 40% across periods (−20%) or Robust CV > 0.8 (−10%).

### Seasonality — STL decomposition

STL (Seasonal-Trend decomposition using LOESS) separates a monthly series into:

```
yₜ = Trendₜ + Seasonalₜ + Residualₜ
```

**Seasonal strength:**

```
Fs = 1 − Var(Residual) / (Var(Seasonal) + Var(Residual))
```

Ranges 0–1. Above 0.3 triggers seasonal mode. Requires ≥ 24 months of data.

STL is computed once per **section** (not per SKU) and the result is inherited by all SKUs in that section. This reduces the number of STL fits from ~133K to ~80, cutting that computation by 99%.

### Stock-out vs Slow Mover

A SKU with zero stock and no recent sales can be in two fundamentally different states:

**Slow Mover** — always had infrequent sales; the absence is normal.

```
Activity Ratio = unique selling days / total span days
Avg Daily Sales = total qty / total span days

Slow Mover if: Activity Ratio < 0.1 AND Avg Daily Sales < 0.5
```

**True Stockout** — was selling consistently, then stopped suddenly.

```
Is Stockout if: current_stock = 0 AND days since last sale > 30
Is True Stockout if also: avg_daily_sales > 1 AND active_days > 30
```

**Lost sales estimation** for confirmed stockouts:

```
Daily rate = pre-stockout qty / pre-stockout period days
Estimated lost qty     = daily rate × days out of stock
Estimated lost revenue = daily rate (value) × days out of stock
Estimated lost profit  = estimated lost revenue × pre-stockout margin
```

### ABC concentration analysis

```
Top-20 contribution = revenue of top 20% SKUs / total section revenue × 100
```

Sections above 80% are flagged **Top-Heavy** — losing one or two SKUs would have an outsized impact on section revenue.

### Status priority

SKUs are assigned a numeric priority (0 = most urgent) so reports sort to the most actionable items first.

| Priority | Status |
|---|---|
| 0 | True Stockout |
| 1 | Potential Stockout (High Value) |
| 2 | Loss-Making |
| 3 | Low Stock (High Margin) |
| 4 | Stockout |
| 5 | Low Stock |
| 6 | Declining (Low Margin) |
| 7 | Declining |
| 8 | Slow Mover |
| 9 | Dead |
| 10 | Inactive |
| 20–26 | Healthy (Growing, Stable, Seasonal…) |

---

## 3 · Pattern & Color Analysis

**Notebook:** `03_pattern_color_analysis.ipynb` · **Module:** `src/pattern_color.py`

### SKU structure

```
SKU:  CCCC PPP CCC ...
      │    │   └── Color  (digits 7–9)
      │    └────── Pattern (digits 4–6)
      └─────────── Collection (digits 0–3)
```

### Performance Score

A single composite metric used to rank patterns and colors:

```
Performance Score = Sales Value + (Total Profit × 0.5) − (Stock Units × 10)
```

The three terms balance three competing objectives:
- **Revenue** (Sales Value) — rewards patterns that actually sell
- **Profitability** (Total Profit × 0.5) — rewards profitable revenue over unprofitable volume
- **Capital efficiency** (Stock × 10) — penalises idle inventory

The penalty coefficient of 10 per unit reflects the opportunity cost of holding one unit of stock. It is intentionally simple; more precise coefficients require a carrying cost estimate.

### Stock-to-sales ratio

```
Stock Sales Ratio = CURRENT_STOCK / Total_Sales_Qty
```

Interpretation:
- **< 0.5** — healthy; less than half a year's sales in stock
- **0.5 – 1.0** — caution; approaching overstock
- **> 1.0** — critical; more stock than has ever been sold in the analysis period

---

## 4 · Demand Forecasting — WMA

**Notebook:** `02_inventory_forecast.ipynb` · **Module:** `src/inventory_forecast.py`

### Why FAMILY only

CONTRACT orders (≥ 100 units) are bulk purchases by institutional buyers. They do not reflect repeatable per-unit retail demand and would inflate the forecast, leading to over-ordering. Separating the two sale types before computing demand is the most important data-quality step in this pipeline.

### Rolling quarters

The 365-day lookback is split into four rolling 90-day windows:

```
Q1: today − 90 days  (most recent)
Q2: today − 180 days
Q3: today − 270 days
Q4: today − 365 days (oldest)
```

Using rolling windows instead of calendar quarters means the forecast always uses the last 12 months of data regardless of where in the year it is run.

### Blended Average Daily Sales (ADS)

Naive ADS = total qty / 90 calendar days would understate demand for a SKU that sells intensively for 20 days then goes out of stock. Equally, it would overstate demand for a SKU sold in a single bulk day.

**Blended denominator:**

```
adjusted_days = α × actual_selling_days + (1−α) × 90
```

Where α = 0.4. This mixes the actual selling-day count with the calendar quarter length:
- SKUs with ≥ 10 selling days in the quarter use the blend
- SKUs with fewer selling days fall back to 90 calendar days

The blended ADS is then capped at `yearly_avg_daily × 1.2` to prevent a single-quarter spike from dominating the forecast.

### Dynamic vs standard weights

**Active SKUs** (≥ 30 total selling days):

```
W_q = share of total annual sales in quarter q
```

With floors: Q1 ≥ 0.30 (recency bias), Q2–Q4 ≥ 0.10.

Normalised to sum to 1.0. These weights automatically capture seasonality — a SKU that sells heavily in Q1 (the most recent period) will have a high Q1 weight and a high forecast.

**Sparse SKUs** (< 30 total selling days):

Fixed weights: Q1=0.40, Q2=0.30, Q3=0.20, Q4=0.10.

The fixed weights are conservative and recency-biased. Sparse SKUs do not have enough data to estimate seasonality reliably.

### Weighted Moving Average

```
ADS_WMA = Q1_ADS × W_Q1 + Q2_ADS × W_Q2 + Q3_ADS × W_Q3 + Q4_ADS × W_Q4
Monthly_WMA = ADS_WMA × 30
```

### Final demand selection

WMA is only used when it exceeds the simple 12-month average:

```
Monthly_Demand_Final = max(Monthly_Actual_Avg, Monthly_WMA)
    — if data coverage ≥ 50% of the lookback window
```

This is a **conservative constraint**: WMA can only upgrade the forecast, never downgrade it. SKUs with insufficient history fall back to the simple average.

### Stockout demand adjustment

If a SKU spent fewer than 120 days with available stock in the last 120 days, its recorded sales understate true demand. The adjustment:

```
factor = (available_days + 0.5 × stockout_days) / available_days
```

Capped at 1.5×. The 0.5 coefficient assumes 50% of demand was permanently lost (not deferred) during the stockout. The cap prevents runaway order quantities for SKUs with extreme stockout histories.

### Reorder calculation

```
Effective_Stock  = CURRENT_STOCK + OUTSTANDING + PR
Target_Stock_7M  = Monthly_Demand_Final × 7
Reorder_Qty      = max(0, Target_Stock_7M − Effective_Stock)
Days_Coverage    = Effective_Stock / ADS_Final
```

The 7-month coverage target accounts for typical replenishment lead times plus a safety buffer. It is intentionally conservative.

---

## 5 · Supplier Analysis

**Notebook:** `05_supplier_analysis.ipynb` · **Module:** `src/supplier_analysis.py`

### What it produces

For a named supplier: a per-SKU table with all yearly sales pivots, stock exposure, margin, activity classification, and (when forecast data is available) reorder quantities. A collection-level summary provides the management view.

### YoY pivot

Annual sales are aggregated and pivoted to produce columns `Sales_YYYY` and `Qty_YYYY` for each year in the data. This allows side-by-side comparison across years without filtering.

### Activity classification

Based on current-year quantity sold and days since last sale:

| Activity | Condition |
|---|---|
| Very Active | Qty > 75th percentile AND last sale < 30 days ago |
| Active | Qty > 0 |
| Slow Moving | 0 < Qty < 25th percentile |
| Inactive | Qty = 0 |

### Stock turnover (revenue-based)

```
Stock_Turnover = Current_Year_Sales / Stock_Value
```

This is a revenue-based proxy because cost-of-goods-sold is not always available at SKU level. It is best used comparatively within the same supplier, not as an absolute benchmark across suppliers with different price points.

### Adjusted cost override

The analysis accepts an optional Excel file with revised unit costs. This handles the case where a supplier's contract has been renegotiated but the new costs have not yet propagated through the ERP system. The adjusted cost file overrides the cost column for matching SKUs; all margin calculations use the adjusted cost.

---

## 6 · Shared Concepts

### Why Parquet instead of CSV

Parquet stores each column in its own compressed segment with the column type preserved. For 4M+ transaction rows:

- CSV: ~1.2 GB, ~45 seconds to load
- Parquet (snappy): ~120 MB, ~3 seconds to load

Column types (dates, integers, nullable booleans) are read back exactly as written, with no re-parsing required.

### Why vectorised operations instead of row loops

Pandas groupby + numpy operations run in C/Cython. A Python `for` loop over 133K SKUs runs at ~2 it/s (17 hours). The same computation expressed as a groupby aggregation runs in seconds because it avoids Python interpreter overhead entirely.

The per-SKU Python loop is retained only where operations genuinely cannot be vectorised — specifically, the monthly trend signal loop, which reads a unique time series per SKU. Even there, all operations inside the loop are numpy (no statsmodels, no Python-level iteration over rows).

### The join safety guarantee

Every enrichment merge in the pipeline checks that the row count before and after is identical. This is enforced by `_left_merge()` in `data_loader.py` and `_safe_left_merge()` in `classification.py`. The check costs one integer comparison and prevents the most common silent data corruption in pandas pipelines: a one-to-many join that multiplies revenue or quantity figures.

---

## 7 · Output Column Reference

### combined_df (produced by 01_data_loading)

| Column | Type | Description |
|---|---|---|
| Date | datetime | Transaction date |
| SKU | string | Product code |
| Section | string | First 4 digits of SKU |
| bal Value | float32 | Transaction revenue (SAR) |
| bal Qty | float32 | Transaction quantity |
| Total_Cost | float32 | Unit cost × qty |
| Total_Profit | float32 | bal Value − Total_Cost |
| Profit_Margin_% | float32 | Total_Profit / bal Value × 100 |
| Sales_Type | string | CONTRACT (≥ 100 units) or FAMILY |
| CURRENT_STOCK | int32 | Units in warehouse |
| OUTSTANDING | int32 | Units on order |
| EFFECTIVE_STOCK | int32 | CURRENT + OUTSTANDING + PR |
| Outlet_Type | string | Retail / Wholesale / Contract |
| Area_Master_Category | string | Geographic area |
| Country | string | Destination country |
| CATEGORY | string | ABC classification (A/B/C/D) |
| Supplier | string | Supplier name |
| ARTPATERN | string | Pattern type (PLAIN / STRIPE / …) |
| IS_PLAIN_SECTION | string | YES if all SKUs in section are PLAIN |
| COLOR_NAME | string | Color description |
| CATALOG_NO | string | Catalog reference (comma-joined if multiple) |

### forecast_df (produced by 02_inventory_forecast)

| Column | Description |
|---|---|
| ADS_WMA | Weighted moving average daily demand |
| Monthly_Demand_Final | Final monthly demand (WMA or actual avg) |
| Forecast_Method | WMA or Actual (Conservative) |
| Weight_Method | Dynamic or Standard |
| Effective_Stock | CURRENT + OUTSTANDING + PR |
| Target_Stock_7M | 7-month coverage target |
| Reorder_Qty | Units to order (0 if sufficient stock) |
| Days_Coverage | Days of stock at current demand rate |
| STATUS | OK / REORDER / CRITICAL / OUT_OF_STOCK / EXCESS |
| Stockout_Adjusted | Yes / No — whether demand was adjusted upward |

### sku_df (produced by 04_sku_collection_classification)

| Column | Description |
|---|---|
| Status | Classification label |
| Key_Reason | Plain-language explanation |
| Priority | 0 = most urgent |
| Revenue_Trend | Growing / Stable / Declining / Unknown |
| Revenue_Confidence | HIGH / MEDIUM / LOW |
| Is_Slow_Mover | True if activity ratio < 0.1 and avg daily < 0.5 |
| Is_True_Stockout | True if sudden stop after consistent sales |
| Is_Potential_Stockout | True if high-rate stop with zero stock |
| Estimated_Lost_Revenue | SAR estimated during stockout |
| Seasonal_Strength | STL seasonal strength 0–1 |
| Trend_Mismatch | True if revenue and quantity trends diverge |
| Top20_Contribution_% | Revenue share of top 20% SKUs (section level) |
