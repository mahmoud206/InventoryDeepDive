# SKU & Section Classification — Methodology

## Overview

The classification engine assigns every SKU and every product section a status label, a priority score, and a plain-language reason.  It is built for high-volatility, log-normal sales distributions typical of fabric and apparel — where bulk orders, seasonal spikes, and stock-outs make standard statistics unreliable.

---

## 1 · Why Standard Statistics Fail on Sales Data

Most sales distributions are **right-skewed and log-normal**, not Gaussian.  A single bulk order can be 50× the median transaction.  This causes:

| Standard metric | Problem |
|---|---|
| Mean | Pulled upward by bulk outliers |
| Standard deviation | Inflated — masks true volatility |
| Pearson correlation | Assumes linearity and normality |

The engine replaces these with **robust alternatives** throughout.

---

## 2 · Robust Statistics (RobustStats)

### Median Absolute Deviation (MAD)

```
MAD = median( |xᵢ − median(x)| )
```

MAD is the median of the absolute deviations from the median.  It ignores extreme values entirely.

### Robust Coefficient of Variation

```
Robust CV = (MAD / median) × 1.4826
```

The **1.4826** scaling factor makes MAD consistent with standard deviation for a Gaussian distribution (it equals 1 / Φ⁻¹(0.75) where Φ is the normal CDF).  This means Robust CV is directly comparable to a classical CV when the data happens to be normal, but remains stable when it is not.

| Robust CV | Interpretation |
|---|---|
| < 0.3 | Low volatility — predictable sales |
| 0.3 – 0.6 | Medium — some variability |
| 0.6 – 1.0 | High — significant spikes |
| > 1.0 | Extreme — normal for seasonal fashion |

### IQR Outlier Detection

```
Lower fence = Q1 − 1.5 × IQR
Upper fence = Q3 + 1.5 × IQR
```

Used to identify bulk orders or data errors.  The 1.5 multiplier is Tukey's original rule; it flags roughly the outermost 0.7% of a normal distribution.

### Winsorisation

Clips values at the 5th and 95th percentiles.  Unlike trimming (which removes points), winsorisation preserves sample size while limiting distortion from extremes.

---

## 3 · Trend Detection — Multi-Signal Voting

No single trend test is reliable on noisy retail data.  The engine uses **four independent signals** and combines them via a weighted vote.

### Signal 1 · Holt's Exponential Smoothing

Fits an additive trend + level model:

```
Level:  Lₜ = α·yₜ + (1−α)(Lₜ₋₁ + Tₜ₋₁)
Trend:  Tₜ = β(Lₜ − Lₜ₋₁) + (1−β)Tₜ₋₁
```

Parameters: α = 0.2 (level smoothing), β = 0.1 (trend smoothing).  Low values mean recent data is weighted over history — appropriate when recent momentum matters more than long-run average.

The **sign and magnitude of the trend component** Tₜ over the last 3 months drives this signal.

### Signal 2 · Mann-Kendall Test

A **non-parametric rank correlation** between time and value:

```
τ = (Concordant pairs − Discordant pairs) / (n(n−1)/2)
```

Advantages:
- Makes no distributional assumptions
- Robust to outliers
- Detects monotonic trends even in non-linear series

The engine applies Mann-Kendall on a **3-point moving average** of the last 12 months — smoothing is applied first to reduce noise amplification.

Threshold: τ > 0.1 and p-value < 0.15 to count as a directional signal.

### Signal 3 · Period-over-Period Change

```
PoPΔ = (mean(last 3 months) − mean(prior 3 months)) / mean(prior 3 months) × 100
```

Simple and interpretable.  Captures near-term momentum without long-run smoothing.

### Signal 4 · Year-over-Year Change (seasonal series only)

For seasonal SKUs (detected via STL — see §4):

```
YoYΔ = (mean(same 3 months last year) − current 3 months) / prior × 100
```

This removes seasonal effects that would otherwise distort the period-over-period signal.

### Weighted Voting

| Signal | Weight | Threshold |
|---|---|---|
| Exponential Smoothing trend | 1.5 | > ±200 units/period |
| Mann-Kendall τ | 2.0 | τ > ±0.1, p < 0.15 |
| Period-over-period Δ | 1.5 | > ±8% |
| Year-over-year Δ (seasonal only) | 2.0 | > ±10% |

Final direction:
- **Growing** if growing-vote share > 60%
- **Declining** if declining-vote share > 60%
- **Stable** otherwise

Confidence is downgraded if:
- Directional consistency < 40% across periods (−20% to confidence score)
- Robust CV > 0.8 (−10%)

---

## 4 · Seasonality Detection — STL Decomposition

Seasonal and Trend decomposition using **LOESS (STL)** separates a time series into:

```
yₜ = Trendₜ + Seasonalₜ + Residualₜ
```

**Seasonal strength** is measured as:

```
Fs = 1 − Var(Residual) / (Var(Seasonal) + Var(Residual))
```

Ranges 0–1.  A value above 0.3 triggers seasonal mode, which:
1. Switches trend detection to YoY comparison instead of period-over-period
2. Labels the SKU/section as `Seasonal` if no stronger status applies

Requires ≥ 24 months of data.  Below that threshold, seasonal strength defaults to 0.

---

## 5 · Stock-out vs Slow Mover — Why the Distinction Matters

A SKU with zero stock and no recent sales can be in two very different states:

| State | Meaning | Action |
|---|---|---|
| **Slow Mover** | Always had infrequent sales — the absence is normal | Review for discontinuation |
| **True Stockout** | Was selling consistently, then stopped suddenly | Reorder immediately |
| **Potential Stockout** | High sales rate + zero stock + extended absence | Escalate urgently |

### Slow Mover Detection

```
Activity Ratio = Unique selling days / Total span days
Avg Daily Sales = Total qty / Total span days

Slow Mover if: Activity Ratio < 0.1  AND  Avg Daily Sales < 0.5
```

### True Stockout Detection

```
Is Stockout if:  current_stock = 0  AND  days since last sale > 30
Is True Stockout if also:  avg_daily_sales > 1  AND  active_days > 30
```

### Lost Sales Estimation

For confirmed stock-outs, the engine estimates what would have been sold during the out-of-stock period:

```
Pre-stockout daily rate (qty)    = pre-stockout qty  / pre-stockout period days
Pre-stockout daily rate (revenue) = pre-stockout rev  / pre-stockout period days

Estimated lost qty     = daily rate (qty)     × days out of stock
Estimated lost revenue = daily rate (revenue) × days out of stock
Estimated lost profit  = estimated lost revenue × (pre-stockout avg margin / 100)
```

The pre-stockout window is defined as all transactions before the last recorded sale date.

---

## 6 · ABC Analysis — Revenue Concentration

For each section:

```
Top-20 contribution = revenue of top 20% SKUs by revenue / total section revenue × 100
```

Sections where > 80% of revenue comes from the top 20% of SKUs are flagged **Top-Heavy**.  This is an operational risk: discontinuing or stocking out one or two SKUs can destroy the section's revenue.

The threshold is based on the Pareto principle, but the 80% line is intentionally strict — a section at 75% is healthy; 85% signals fragility.

---

## 7 · Trend Mismatch Signals

Revenue and quantity trends are computed independently.  When they diverge (and both have sufficient confidence):

| Revenue | Quantity | Interpretation |
|---|---|---|
| Growing | Declining | Price increase or premium product mix shift |
| Declining | Growing | Discounting or shift to lower-value products |
| Stable | Growing | Volume growth without revenue capture |
| Stable | Declining | Volume erosion without revenue impact |

These mismatches do not override the primary status but are appended to `Key_Reason` and flagged in `Trend_Mismatch`.

---

## 8 · Inventory Turnover

At the section level:

```
Inventory Turnover Ratio = 12-month Revenue / Current Stock Value
```

This is a revenue-based proxy (not cost-based) due to data availability.  Interpretation is relative, not absolute:

- **< 2×** — slow-moving stock, potential overstock
- **4–8×** — healthy range for most product categories
- **> 12×** — fast-moving; stock-out risk if lead times are long

---

## 9 · Status Priority Hierarchy

SKUs are assigned a numeric priority (0 = most urgent) to enable sorted reporting:

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
| 20–26 | Healthy statuses (Growing, Stable, Seasonal…) |

---

## 10 · Output Columns Reference

### SKU-level

| Column | Description |
|---|---|
| `Status` | Assigned classification label |
| `Key_Reason` | Plain-language explanation |
| `Priority` | Numeric urgency (0 = highest) |
| `Revenue_Trend` | Growing / Stable / Declining / Unknown |
| `Revenue_Confidence` | HIGH / MEDIUM / LOW |
| `Revenue_Period_Change_%` | Last 3M vs prior 3M % change |
| `Revenue_YoY_Change_%` | YoY % change (seasonal only) |
| `Revenue_Robust_CV` | Robust coefficient of variation |
| `Is_Slow_Mover` | True if activity ratio < 0.1 and avg daily < 0.5 |
| `Is_True_Stockout` | True if sudden stop after consistent sales |
| `Is_Potential_Stockout` | True if high-rate stop with zero stock |
| `Estimated_Lost_Revenue` | SAR estimated during stockout period |
| `Estimated_Lost_Profit` | SAR estimated profit loss |
| `Inventory_Turnover_Days` | Days of stock at current sales rate |
| `Seasonal_Strength` | STL seasonal strength 0–1 |
| `Trend_Mismatch` | True if revenue and quantity diverge |

### Section-level

| Column | Description |
|---|---|
| `Pct_Growing` | % of SKUs with Growing trend |
| `Pct_Declining` | % of SKUs with Declining trend |
| `High_Conf_Growing` | SKUs with HIGH confidence Growing |
| `Top20_Contribution_%` | Revenue share of top 20% SKUs |
| `Is_Top_Heavy` | True if Top20 > 80% |
| `Inventory_Turnover_Ratio` | 12M Revenue / Stock |
| `True_Stockouts` | Count of True Stockout SKUs in section |
| `Slow_Movers` | Count of Slow Mover SKUs in section |
