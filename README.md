# Inventory Deep Dive

A production-grade supply chain analytics system built for a fabric and apparel distributor operating across Saudi Arabia and the GCC. The project covers the full analytical pipeline — from raw ERP exports to actionable purchase decisions — across four interconnected notebooks.

---

## The Problem

A distributor with **133,000+ SKUs**, **4.2M+ transactions**, and **3000+ product sections** needed answers to four operational questions:

1. How do we build a clean, enriched dataset from fragmented yearly CSV exports and 13 auxiliary files — without corrupting margins, duplicating rows, or losing historical SKUs?
2. Which SKUs are growing, declining, stocking out, or quietly dying — and what should we do about each one?
3. How many units of each SKU should we reorder, and when?
4. How is each supplier actually performing — revenue, margin, stock exposure, and activity — broken down by collection and year?

Each notebook answers one question. Each builds directly on the output of the previous one.

---

## Project Structure

```
inventory-analysis/
│
├── notebooks/
│   ├── 01_data_loading.ipynb               ← Build combined_df
│   ├── 02_sku_collection_classification.ipynb  ← Classify every SKU
│   ├── 03_inventory_forecast.ipynb         ← Calculate reorder quantities
│   └── 04_supplier_analysis.ipynb          ← Supplier performance report
│
├── src/
│   ├── data_loader.py          ← Ingestion + enrichment pipeline
│   ├── classification.py       ← SKU/section classification engine
│   ├── inventory_forecast.py   ← WMA demand forecast
│   └── supplier_analysis.py    ← Supplier analytics
│
├── outputs/                    ← Parquet + Excel outputs (gitignored)
├── generate_sample_data.py     ← Synthetic data generator for demo
├── METHODOLOGY.md              ← Full mathematical reference
├── requirements.txt
└── README.md
```

---

## How the Notebooks Connect

The four notebooks form a sequential pipeline. Each one produces an output that the next one consumes.

```
01_data_loading
      │
      │  combined_df.parquet
      │  (4.2M rows · 40+ columns · enriched transactions)
      │
      ├──────────────────────────────────────────┐
      │                                          │
      ▼                                          ▼
02_sku_collection_classification      03_inventory_forecast
      │                                          │
      │  sku_df                                  │  forecast_df
      │  (status, trend, stockout,               │  (reorder qty,
      │   lost revenue per SKU)                  │   days coverage,
      │                                          │   WMA demand)
      └──────────────────┬───────────────────────┘
                         │
                         │  Both feed into
                         ▼
                04_supplier_analysis
                         │
                         │  per-supplier Excel report
                         │  (YoY revenue, margin,
                         │   stock exposure, reorder list)
```

`combined_df.parquet` is the single source of truth. Run `01` once. Every other notebook loads from the Parquet file — cold start to full analysis in seconds, not minutes.

---

## Notebooks

### 01 · Data Loading

**Input:** Raw sales CSVs (yearly) + 13 auxiliary Excel files (costs, stock levels, categories, suppliers, outlets, catalogs, colors, patterns)

**Output:** `combined_df.parquet`

**What it does:**
- Loads all yearly CSV files, deduplicates, sorts, and concatenates
- Filters internal/non-commercial outlets by keyword match
- Assigns year-matched unit costs to every transaction row (a 2022 sale uses 2022 costs, not current ones)
- Calculates `Total_Cost`, `Total_Profit`, `Profit_Margin_%` per transaction
- Left-joins all 13 enrichment files with a row-count guard — if any join would inflate rows, the pipeline raises immediately rather than silently corrupting margin figures
- Outer-joins pre-2019 historical quantity data to preserve legacy SKUs
- Classifies transactions as CONTRACT (≥ 100 units) or FAMILY

The pipeline can be run as a script or imported:
```python
from src.data_loader import run
combined_df = run()

# All subsequent notebooks load from Parquet in ~3 seconds
combined_df = pd.read_parquet("outputs/combined_df.parquet")
```

---

### 02 · SKU & Section Classification

**Input:** `combined_df.parquet`

**Output:** `sku_df` (one row per SKU), `section_df` (one row per section), `FABRIC_ANALYSIS_COMPLETE.xlsx`

**What it does:**

Assigns every SKU a status label, priority score, and plain-language reason using a multi-signal classification engine.

**Trend detection** uses four independent signals with weighted voting:

| Signal | Method | Weight |
|--------|--------|--------|
| Recent momentum | Holt double exponential smoothing (numpy EMA) | 1.5 |
| Monotonic trend | Spearman rank correlation | 2.0 |
| Near-term change | Last 3M vs prior 3M % change | 1.5 |
| Seasonal change | Year-over-year % (seasonal SKUs only) | 2.0 |

**Stockout vs Slow Mover** — a SKU with zero stock and no recent sales is either a slow mover (always had low activity) or a true stockout (was selling well, then stopped). The classification distinguishes these explicitly. True stockouts include an estimated lost revenue calculation.

**Seasonality** is detected via STL decomposition, computed once per section (~80 computations) and inherited by all SKUs in that section — instead of running STL 133K times per SKU.

**Performance:** 133K SKUs in ~8 minutes on a standard laptop. The original implementation required ~17 hours.

Key optimisations:
- All scalar metrics computed via `groupby` (single vectorised pass, no per-SKU slice)
- `ExponentialSmoothing` (statsmodels) → fast numpy Holt implementation (~100× per call)
- `kendalltau` (O(n²)) → `spearmanr` (O(n log n)) for trend testing
- `np.select` for status assignment (no row-wise `apply`)

---

### 03 · Inventory Demand Forecast

**Input:** `combined_df.parquet`

**Output:** `forecast_df` (one row per SKU), `Demand_Forecast_YYYYMMDD.xlsx`

**What it does:**

Calculates a per-SKU reorder quantity using a Weighted Moving Average (WMA) built on four rolling 90-day quarters.

**Why FAMILY only:** CONTRACT bulk orders (≥ 100 units) do not represent repeatable retail demand. Including them would inflate the forecast and cause systematic over-ordering.

**Rolling quarters** (Q1 = most recent):

```
Q1: last 90 days    Q2: 91–180 days
Q3: 181–270 days    Q4: 271–365 days
```

**Blended Average Daily Sales** prevents demand inflation for intermittent SKUs:

```
adjusted_days = 0.4 × actual_selling_days + 0.6 × 90
ADS_q = quarterly_qty / adjusted_days    (capped at 1.2× yearly average)
```

**Dynamic weights** for active SKUs (≥ 30 selling days total):

```
W_q = share of annual sales in quarter q    (Q1 floor: 30%, others: 10%)
```

Fixed weights (40/30/20/10) for sparse SKUs. Dynamic weights automatically capture seasonality without a separate seasonal model.

**Conservative forecast selection:** WMA is only used when it exceeds the simple 12-month average. WMA can upgrade the forecast; it never downgrades it.

**Stockout demand adjustment:** For SKUs that experienced stockout gaps, demand is adjusted upward:

```
factor = (available_days + 0.5 × stockout_days) / available_days    (cap: 1.5×)
```

The 0.5 coefficient assumes 50% of demand during the stockout was permanently lost.

**Reorder calculation:**

```
Target_Stock_7M = Monthly_Demand_Final × 7
Reorder_Qty     = max(0, Target − Effective_Stock)
```

No visuals — this notebook is an operational tool, not an exploration. Output is a number that goes into a purchase order.

---

### 04 · Supplier Analysis

**Input:** `combined_df.parquet` + optionally `forecast_df` from notebook 03

**Output:** Per-supplier `sku_df`, `collection_df`, `Supplier_<name>_YYYYMMDD.xlsx`

**What it does:**

Produces a full supplier performance report: YoY revenue by collection, margin by SKU, stock exposure, activity classification, and (when forecast data is provided) reorder requirements.

**YoY pivot:** Annual sales are pivoted to columns `Sales_YYYY` and `Qty_YYYY` for direct year-by-year comparison without filtering.

**Activity classification:**

| Level | Condition |
|-------|-----------|
| Very Active | Current-year qty > 75th percentile AND last sale < 30 days ago |
| Active | Current-year qty > 0 |
| Slow Moving | 0 < qty < 25th percentile |
| Inactive | No current-year sales |

**Stock turnover (revenue-based):**

```
Stock_Turnover = Current_Year_Sales / Stock_Value
```

**Adjusted cost override:** Accepts an optional Excel file with renegotiated unit costs. When a supplier contract changes, pass the new cost file to recalculate all margins without modifying `combined_df`.

**Linking to notebook 03:** When `forecast_df` is passed, reorder quantities are merged onto the SKU table. The supplier report then shows not just how each collection is performing, but exactly how many units need to be ordered from that supplier in the next purchase cycle.

Five static Plotly charts render directly in GitHub's notebook viewer:
- Monthly revenue timeline (dual-axis with quantity)
- YoY collection comparison
- SKU portfolio map (margin vs turnover, sized by revenue)
- Activity breakdown by collection
- Stock exposure map (stock value vs sales ratio)

---

## Running with Synthetic Data

Real data is not included in this repository. To run all notebooks end-to-end with realistic synthetic data:

```bash
python generate_sample_data.py
```

This writes `outputs/combined_df.parquet` (~40 MB) with the same schema as the real dataset: 200K transactions, 2,000 SKUs, 80 sections, 150 outlets, 5 years of history. All notebooks run without modification.

---

## Setup

```bash
git clone https://github.com/<your-username>/inventory-deep-dive.git
cd inventory-deep-dive

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt

python generate_sample_data.py   # create synthetic combined_df.parquet
jupyter notebook
```

**requirements.txt**

```
pandas>=2.0
numpy>=1.24
scipy>=1.11
statsmodels>=0.14
plotly>=5.18
openpyxl>=3.1
pyarrow>=14.0
tqdm>=4.66
```

---

## Key Design Principles

**No silent data corruption.** Every enrichment join in the pipeline asserts that row count is unchanged after the merge. If a join key has duplicates in an enrichment file, the pipeline raises immediately. One duplicate in an outlet file could double revenue figures for half the dataset; this check costs one integer comparison.

**Conservative over optimistic.** The WMA forecast only upgrades demand, never downgrades it. The reorder calculation uses a 7-month coverage target. The stockout adjustment caps at 1.5×. Every assumption errs toward ordering less rather than more.

**Separation of execution and exploration.** Notebooks 01 and 03 are execution tools — they produce a number or a file. Notebooks 02 and 04 are exploration tools — they produce charts and analytical context. The distinction determines whether visuals are included.

**Reproducibility without real data.** The synthetic generator replicates the full schema, realistic value distributions, and seasonal patterns. Any reviewer can clone this repository and reproduce every chart and table in under five minutes.

---

## Methodology

Full mathematical reference — all formulas, design decisions, and output column definitions — is documented in [`METHODOLOGY.md`](METHODOLOGY.md).


