"""
classification.py  ·  src/
============================
SKU and Section classification engine — vectorised for large datasets.

Performance vs naive per-SKU loop:
    ExponentialSmoothing (statsmodels) → fast numpy Holt EMA  (~100x per call)
    Mann-Kendall (kendalltau)          → Spearman rho (scipy) (~100x per call)
    Per-SKU loop                       → groupby vectorisation for all scalar metrics
    STL seasonality                    → computed once per section, reused per SKU

Result: 133K SKUs in ~8 minutes instead of ~17 hours.

Usage
-----
    from src.classification import ultimate_fabric_analysis
    sku_df, section_df = ultimate_fabric_analysis(combined_df)
"""

import gc
import warnings
from datetime import timedelta

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from statsmodels.tsa.seasonal import STL
from tqdm import tqdm

warnings.filterwarnings("ignore")


# ── Signal functions ──────────────────────────────────────────────────────────

def _robust_cv(values: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    v = np.asarray(values, dtype=np.float64)
    med = np.nanmedian(v)
    if med == 0 or np.isnan(med):
        return 1.0 if np.nanmax(v) > 0 else 0.0
    return float(min(np.nanmedian(np.abs(v - med)) / med * 1.4826, 10.0))


def _ema_trend(values: np.ndarray, alpha: float = 0.2, beta: float = 0.1) -> float:
    """Holt double-exponential smoothing — pure numpy, no statsmodels."""
    if len(values) < 4:
        return 0.0
    v = np.asarray(values, dtype=np.float64)
    L, T = v[0], v[1] - v[0]
    trends = []
    for x in v[1:]:
        L, T = alpha * x + (1 - alpha) * (L + T), beta * (alpha * x + (1 - alpha) * (L + T) - L) + (1 - beta) * T
        trends.append(T)
    return float(np.mean(trends[-3:])) if len(trends) >= 3 else float(trends[-1])


def _spearman_trend(values: np.ndarray) -> tuple:
    """Spearman rank correlation vs time. Replaces Mann-Kendall (~100x faster)."""
    if len(values) < 4:
        return 0.0, 1.0
    try:
        rho, p = spearmanr(np.arange(len(values)), values)
        return (float(rho) if not np.isnan(rho) else 0.0,
                float(p)   if not np.isnan(p)   else 1.0)
    except Exception:
        return 0.0, 1.0


def _period_change(v: np.ndarray) -> float:
    if len(v) < 6:
        return 0.0
    prev = v[-6:-3].mean()
    return float((v[-3:].mean() - prev) / prev * 100) if prev > 0 else 0.0


def _yoy_change(v: np.ndarray):
    if len(v) < 24:
        return None
    prev = v[-15:-12].mean()
    return float((v[-3:].mean() - prev) / prev * 100) if prev > 0 else None


def _seasonal_strength(v: np.ndarray) -> float:
    if len(v) < 24:
        return 0.0
    try:
        res = STL(v.astype(float), period=12, robust=True).fit()
        sv, rv = np.var(res.seasonal), np.var(res.resid)
        return float(np.clip(1 - rv / (sv + rv), 0, 1)) if sv + rv > 0 else 0.0
    except Exception:
        return 0.0


def _vote(ema_t, rho, p, pop, yoy, cv, consistency) -> tuple:
    vg = vd = tw = 0.0
    if   ema_t < -200: vd += 1.5; tw += 1.5
    elif ema_t >  200: vg += 1.5; tw += 1.5
    if   rho < -0.1 and p < 0.15: vd += 2; tw += 2
    elif rho >  0.1 and p < 0.15: vg += 2; tw += 2
    if   pop < -8: vd += 1.5; tw += 1.5
    elif pop >  8: vg += 1.5; tw += 1.5
    if yoy is not None:
        if   yoy < -10: vd += 2; tw += 2
        elif yoy >  10: vg += 2; tw += 2
    if tw == 0:
        d, cs = 'Stable', 0.5
    else:
        gs, ds = vg / tw, vd / tw
        if   gs > 0.6: d, cs = 'Growing',  gs
        elif ds > 0.6: d, cs = 'Declining', ds
        else:          d, cs = 'Stable',   1 - abs(gs - ds)
    cs = min(cs * 1.2, 1.0) if consistency > 70 else (cs * 0.8 if consistency < 40 else cs)
    if cv > 0.8:
        cs *= 0.9
    return d, ('HIGH' if cs >= 0.7 else 'MEDIUM' if cs >= 0.5 else 'LOW')


# ── Vectorised scalar metrics ─────────────────────────────────────────────────

def _scalar_metrics(df: pd.DataFrame, analysis_date: pd.Timestamp) -> pd.DataFrame:
    one_year_ago = analysis_date - timedelta(days=365)

    agg = df.groupby('SKU_CLEAN', sort=False).agg(
        Total_Revenue = ('bal Value', 'sum'),
        Total_Qty     = ('bal Qty',   'sum'),
        Total_Txns    = ('bal Value', 'count'),
        First_Sale    = ('Date',      'min'),
        Last_Sale     = ('Date',      'max'),
        Section       = ('Section',   'first') if 'Section' in df.columns else ('SKU_CLEAN', 'first'),
    )

    if 'Total_Profit' in df.columns:
        agg = agg.join(df.groupby('SKU_CLEAN', sort=False)['Total_Profit'].sum())
    else:
        agg['Total_Profit'] = 0.0

    if 'CURRENT_STOCK' in df.columns:
        agg = agg.join(df.groupby('SKU_CLEAN', sort=False)['CURRENT_STOCK'].first().rename('Current_Stock'))
    else:
        agg['Current_Stock'] = 0.0

    recent = df[df['Date'] >= one_year_ago]
    if len(recent):
        ra = recent.groupby('SKU_CLEAN', sort=False).agg(
            Recent_Revenue = ('bal Value', 'sum'),
            Recent_Qty     = ('bal Qty',   'sum'),
            Recent_Txns    = ('bal Value', 'count'),
        )
        if 'Total_Profit' in recent.columns:
            ra = ra.join(recent.groupby('SKU_CLEAN', sort=False)['Total_Profit'].sum().rename('Recent_Profit'))
        else:
            ra['Recent_Profit'] = 0.0
        agg = agg.join(ra, how='left')
    agg = agg.fillna({'Recent_Revenue': 0, 'Recent_Qty': 0, 'Recent_Txns': 0, 'Recent_Profit': 0})

    agg['Days_Since_Last']  = (analysis_date - agg['Last_Sale']).dt.days
    agg['Product_Age_Days'] = (analysis_date - agg['First_Sale']).dt.days
    agg['Avg_Margin_%']        = np.where(agg['Total_Revenue']  > 0, agg['Total_Profit']  / agg['Total_Revenue']  * 100, 0)
    agg['Recent_Avg_Margin_%'] = np.where(agg['Recent_Revenue'] > 0, agg['Recent_Profit'] / agg['Recent_Revenue'] * 100, 0)

    active = df.groupby('SKU_CLEAN', sort=False)['Date'].nunique()
    agg    = agg.join(active.rename('Active_Days'))
    agg['Activity_Ratio']         = agg['Active_Days'] / agg['Product_Age_Days'].clip(lower=1)
    agg['Avg_Daily_Sales']        = agg['Total_Qty']   / agg['Product_Age_Days'].clip(lower=1)
    agg['Is_Slow_Mover']          = (agg['Activity_Ratio'] < 0.1) & (agg['Avg_Daily_Sales'] < 0.5)
    agg['Is_Stockout']            = (agg['Current_Stock'] == 0) & (agg['Days_Since_Last'] > 30)
    agg['Is_True_Stockout']       = agg['Is_Stockout'] & (agg['Avg_Daily_Sales'] > 1) & (agg['Active_Days'] > 30) & ~agg['Is_Slow_Mover']
    agg['Inventory_Turnover_Days']= np.where((agg['Avg_Daily_Sales'] > 0) & (agg['Current_Stock'] > 0),
                                              agg['Current_Stock'] / agg['Avg_Daily_Sales'], 0)
    return agg.reset_index()


# ── Monthly trend signals ─────────────────────────────────────────────────────

def _monthly_trends(df: pd.DataFrame, sec_seasonal: dict,
                    analysis_date: pd.Timestamp) -> pd.DataFrame:
    df2 = df.assign(YM=df['Date'].dt.to_period('M'))
    pivot_r = df2.groupby(['SKU_CLEAN', 'YM'], sort=False)['bal Value'].sum().unstack('SKU_CLEAN', fill_value=0)
    pivot_q = df2.groupby(['SKU_CLEAN', 'YM'], sort=False)['bal Qty'].sum().unstack('SKU_CLEAN', fill_value=0)

    records = []
    for sku in tqdm(pivot_r.columns, desc="SKUs", ncols=80):
        mr, mq = pivot_r[sku].values, pivot_q[sku].values
        if len(mr) < 6:
            records.append({'SKU_CLEAN': sku, 'Revenue_Trend': 'Unknown',
                             'Revenue_Confidence': 'LOW', 'Revenue_Period_Change_%': 0,
                             'Revenue_YoY_Change_%': 0, 'Revenue_Robust_CV': 0,
                             'Quantity_Trend': 'Unknown', 'Quantity_Confidence': 'LOW',
                             'Quantity_Period_Change_%': 0, 'Quantity_YoY_Change_%': 0,
                             'Quantity_Robust_CV': 0, 'Is_Seasonal': False,
                             'Seasonal_Strength': 0.0, 'Active_Months': len(mr),
                             'Avg_Monthly_Revenue': float(mr.mean()),
                             'Trend_Mismatch': False, 'Mismatch_Warning': None})
            continue

        sec      = df2.loc[df2['SKU_CLEAN'] == sku, 'Section'].iloc[0] if 'Section' in df2.columns else 'Unknown'
        seas     = sec_seasonal.get(str(sec), 0.0)
        is_s     = seas > 0.3
        tail     = lambda v: v[-12:] if len(v) > 12 else v
        cons     = lambda v: (max((np.diff(tail(v)) > 0).sum(), (np.diff(tail(v)) < 0).sum()) / max(len(np.diff(tail(v))), 1) * 100)

        rd, rc = _vote(_ema_trend(mr), *_spearman_trend(tail(mr)), _period_change(mr),
                       _yoy_change(mr) if is_s else None, _robust_cv(mr), cons(mr))
        qd, qc = _vote(_ema_trend(mq), *_spearman_trend(tail(mq)), _period_change(mq),
                       _yoy_change(mq) if is_s else None, _robust_cv(mq), cons(mq))

        mismatch = rd != qd and rc != 'LOW' and qc != 'LOW'
        warn = None
        if mismatch:
            if   rd == 'Growing'  and qd == 'Declining': warn = "Revenue UP / Qty DOWN → price increase or mix shift"
            elif rd == 'Declining' and qd == 'Growing':  warn = "Revenue DOWN / Qty UP → discounting"
            elif rd == 'Stable':                          warn = f"Revenue stable / Qty {qd}"
            elif qd == 'Stable':                          warn = f"Qty stable / Revenue {rd}"

        records.append({'SKU_CLEAN': sku,
            'Revenue_Trend': rd, 'Revenue_Confidence': rc,
            'Revenue_Period_Change_%': float(_period_change(mr)),
            'Revenue_YoY_Change_%': float(_yoy_change(mr) or 0),
            'Revenue_Robust_CV': float(_robust_cv(mr)),
            'Quantity_Trend': qd, 'Quantity_Confidence': qc,
            'Quantity_Period_Change_%': float(_period_change(mq)),
            'Quantity_YoY_Change_%': float(_yoy_change(mq) or 0),
            'Quantity_Robust_CV': float(_robust_cv(mq)),
            'Is_Seasonal': is_s, 'Seasonal_Strength': float(seas),
            'Active_Months': len(mr), 'Avg_Monthly_Revenue': float(mr.mean()),
            'Trend_Mismatch': mismatch, 'Mismatch_Warning': warn})

    return pd.DataFrame(records)


# ── Lost-sales estimation (stockout SKUs only) ────────────────────────────────

def _lost_sales(df: pd.DataFrame, sku_df: pd.DataFrame,
                analysis_date: pd.Timestamp, has_profit: bool) -> pd.DataFrame:
    so_skus = sku_df.loc[sku_df['Is_Stockout'], 'SKU_CLEAN'].tolist()
    defaults = {'Estimated_Lost_Sales_Qty': 0.0, 'Estimated_Lost_Revenue': 0.0,
                'Estimated_Lost_Profit': 0.0, 'Is_Potential_Stockout': False, 'Days_Out_of_Stock': 0}
    if not so_skus:
        return sku_df.assign(**defaults)

    rows = []
    for sku, grp in df[df['SKU_CLEAN'].isin(so_skus)].groupby('SKU_CLEAN', sort=False):
        dates     = pd.to_datetime(grp['Date'].values)
        last_sale = dates.max()
        days_out  = (analysis_date - last_sale).days
        pre       = dates < last_sale
        lq = lr = lp = 0.0
        is_pot = False
        if pre.any():
            period  = max((last_sale - dates[pre].min()).days, 1)
            rate_q  = grp['bal Qty'].values[pre].sum()   / period
            rate_v  = grp['bal Value'].values[pre].sum() / period
            lq, lr  = rate_q * days_out, rate_v * days_out
            is_pot  = (rate_q > 2 or rate_v > 2000) and len(np.unique(dates[pre])) >= 30
            if has_profit and 'Total_Profit' in grp.columns:
                pre_rev = grp['bal Value'].values[pre].sum()
                if pre_rev > 0:
                    margin = grp['Total_Profit'].values[pre].sum() / pre_rev
                    lp     = lr * margin
        rows.append({'SKU_CLEAN': sku, 'Estimated_Lost_Sales_Qty': lq,
                     'Estimated_Lost_Revenue': lr, 'Estimated_Lost_Profit': lp,
                     'Is_Potential_Stockout': is_pot, 'Days_Out_of_Stock': days_out})

    lost = pd.DataFrame(rows)
    sku_df = sku_df.merge(lost, on='SKU_CLEAN', how='left')
    for k, v in defaults.items():
        sku_df[k] = sku_df[k].fillna(v)
    return sku_df


# ── Status assignment (vectorised np.select) ──────────────────────────────────

_PRIORITY = {
    "True Stockout": 0, "Potential Stockout (High Value)": 1, "Loss-Making": 2,
    "Low Stock (High Margin)": 3, "Stockout": 4, "Low Stock": 5,
    "Declining (Low Margin)": 6, "Declining": 7, "Slow Mover": 8,
    "Dead": 9, "Inactive": 10, "Growing (High Margin)": 20, "Growing": 21,
    "Stable (Profitable)": 22, "Stable": 23, "Seasonal": 24, "New": 25,
}


def _b(s) -> np.ndarray:
    """Convert any boolean-like Series to a plain numpy bool array for np.select."""
    return np.asarray(s, dtype=bool)


def _assign_sku_status(df: pd.DataFrame) -> pd.DataFrame:
    m    = df
    c_ts   = _b(m['Is_True_Stockout'])
    c_ps   = _b(m.get('Is_Potential_Stockout', pd.Series(False, index=m.index)))
    c_so   = _b(m['Is_Stockout']) & ~c_ts & ~c_ps
    c_lm   = ~c_ts & ~c_ps & ~c_so & _b(m['Recent_Avg_Margin_%'] < 0)
    c_lshm = ~c_lm & _b(m['Current_Stock'].between(1, 9)) & _b(m['Recent_Avg_Margin_%'] > 30)
    c_ls   = ~c_lm & ~c_lshm & _b(m['Current_Stock'].between(1, 9))
    c_sl   = _b(m['Is_Slow_Mover']) & ~c_ts & ~c_ps & ~c_so & ~c_lm
    c_dead = ~c_sl & _b(m['Days_Since_Last'] > 365)
    c_inac = ~c_sl & ~c_dead & _b(m['Days_Since_Last'] > 180)
    c_new  = ~c_sl & ~c_dead & ~c_inac & _b(m['Recent_Txns'] < 3) & _b(m['Days_Since_Last'] < 90)
    c_dlm  = _b(m['Revenue_Trend'] == 'Declining') & _b(m['Recent_Avg_Margin_%'] < 15)
    c_dec  = _b(m['Revenue_Trend'] == 'Declining') & ~c_dlm
    c_ghm  = _b(m['Revenue_Trend'] == 'Growing')   & _b(m['Recent_Avg_Margin_%'] > 30)
    c_grow = _b(m['Revenue_Trend'] == 'Growing')   & ~c_ghm
    c_seas = _b(m.get('Is_Seasonal', pd.Series(False, index=m.index)))
    c_stp  = _b(m['Revenue_Trend'] == 'Stable')    & _b(m['Recent_Avg_Margin_%'] > 25)

    conds  = [c_ts, c_ps, c_so, c_lm, c_lshm, c_ls, c_sl, c_dead, c_inac, c_new,
              c_dlm, c_dec, c_ghm, c_grow, c_seas, c_stp]
    labels = ["True Stockout", "Potential Stockout (High Value)", "Stockout",
              "Loss-Making", "Low Stock (High Margin)", "Low Stock", "Slow Mover",
              "Dead", "Inactive", "New", "Declining (Low Margin)", "Declining",
              "Growing (High Margin)", "Growing", "Seasonal", "Stable (Profitable)"]

    df = df.copy()
    df['Status']   = np.select(conds, labels, default='Stable')
    df['Priority'] = df['Status'].map(_PRIORITY).fillna(99).astype(int)

    ma = df['Recent_Avg_Margin_%'].round(1).astype(str)
    st = df['Current_Stock'].fillna(0).astype(int).astype(str)
    ds = df['Days_Since_Last'].astype(str)
    mw = df.get('Mismatch_Warning', pd.Series('', index=df.index)).fillna('')

    reason = np.select(conds, [
        "Sudden stop after consistent sales",
        "High sales rate + zero stock",
        ds + " days no stock",
        "Recent margin " + ma + "%",
        st + " units + " + ma + "% margin",
        st + " units remaining",
        "Activity ratio < 10%",
        "No sales > 1 year",
        ds + " days since last sale",
        "Under evaluation",
        "Revenue down + " + ma + "% margin",
        "Revenue trending down",
        "Revenue up + " + ma + "% margin",
        "Revenue trending up",
        "Seasonal pattern detected",
        "Steady + " + ma + "% margin",
    ], default="Steady performance")

    df['Key_Reason'] = np.where(df.get('Trend_Mismatch', False) & (mw != ''),
                                 reason + " | " + mw, reason)
    return df


# ── Section analysis ──────────────────────────────────────────────────────────

def _section_analysis(df: pd.DataFrame, sku_df: pd.DataFrame,
                       analysis_date: pd.Timestamp) -> pd.DataFrame:
    has_profit = 'Total_Profit' in df.columns
    has_stock  = 'CURRENT_STOCK' in df.columns
    one_year_ago = analysis_date - timedelta(days=365)
    df2 = df.assign(YM=df['Date'].dt.to_period('M'))

    sec = df2.groupby('Section', sort=False).agg(
        Total_Revenue = ('bal Value', 'sum'),
        Total_Qty     = ('bal Qty',   'sum'),
        Num_Txns      = ('bal Value', 'count'),
        First_Sale    = ('Date',      'min'),
        Last_Sale     = ('Date',      'max'),
        Total_SKUs    = ('SKU',       'nunique'),
    )
    if has_profit:
        sec = sec.join(df2.groupby('Section', sort=False)['Total_Profit'].sum())
    else:
        sec['Total_Profit'] = 0.0

    rec = df2[df2['Date'] >= one_year_ago]
    ra  = rec.groupby('Section', sort=False).agg(
        Recent_Revenue = ('bal Value', 'sum'),
        Recent_Qty     = ('bal Qty',   'sum'),
    )
    if has_profit:
        ra = ra.join(rec.groupby('Section', sort=False)['Total_Profit'].sum().rename('Recent_Profit'))
    else:
        ra['Recent_Profit'] = 0.0
    sec = sec.join(ra, how='left').fillna({'Recent_Revenue': 0, 'Recent_Qty': 0, 'Recent_Profit': 0})

    if has_stock:
        sv    = df2.groupby(['Section', 'SKU'], sort=False)['CURRENT_STOCK'].first()
        sec   = sec.join(sv.groupby('Section').sum().rename('Total_Stock'))
        sec   = sec.join((sv == 0).groupby('Section').sum().rename('SKUs_Out_of_Stock'))
        sec   = sec.join(sv.between(1, 9).groupby('Section').sum().rename('SKUs_Low_Stock'))
        sec['Stock_Out_Rate_%'] = sec['SKUs_Out_of_Stock'] / sec['Total_SKUs'] * 100
    else:
        sec[['Total_Stock', 'SKUs_Out_of_Stock', 'SKUs_Low_Stock', 'Stock_Out_Rate_%']] = 0.0

    sec['Days_Since_Last_Sale']    = (analysis_date - sec['Last_Sale']).dt.days
    sec['Avg_Margin_%']            = np.where(sec['Total_Revenue']  > 0, sec['Total_Profit']  / sec['Total_Revenue']  * 100, 0)
    sec['Recent_Avg_Margin_%']     = np.where(sec['Recent_Revenue'] > 0, sec['Recent_Profit'] / sec['Recent_Revenue'] * 100, 0)
    sec['Inventory_Turnover_Ratio']= np.where((sec.get('Total_Stock', 0) > 0) & (sec['Recent_Revenue'] > 0),
                                               sec['Recent_Revenue'] / sec.get('Total_Stock', pd.Series(1, index=sec.index)), 0)

    sku_col = 'SKU' if 'SKU' in sku_df.columns else 'SKU_CLEAN'
    if 'Section' in sku_df.columns and len(sku_df):
        h = sku_df.groupby('Section', sort=False).agg(
            SKUs_Growing        = ('Revenue_Trend', lambda x: (x == 'Growing').sum()),
            SKUs_Declining      = ('Revenue_Trend', lambda x: (x == 'Declining').sum()),
            SKUs_Stable         = ('Revenue_Trend', lambda x: (x == 'Stable').sum()),
            Slow_Movers         = ('Is_Slow_Mover', 'sum'),
            True_Stockouts      = ('Is_True_Stockout', 'sum'),
        )
        if 'Is_Potential_Stockout' in sku_df.columns:
            h = h.join(sku_df.groupby('Section')['Is_Potential_Stockout'].sum().rename('Potential_Stockouts'))
        sec = sec.join(h, how='left').fillna(0)
        sec['Pct_Growing']   = sec['SKUs_Growing']   / sec['Total_SKUs'] * 100
        sec['Pct_Declining'] = sec['SKUs_Declining'] / sec['Total_SKUs'] * 100
    else:
        for c in ['SKUs_Growing','SKUs_Declining','SKUs_Stable','Slow_Movers',
                  'True_Stockouts','Potential_Stockouts','Pct_Growing','Pct_Declining']:
            sec[c] = 0

    sku_rev  = df2.groupby(['Section', 'SKU'], sort=False)['bal Value'].sum()
    top20    = sku_rev.groupby('Section').apply(
        lambda x: x.nlargest(max(1, int(len(x) * 0.2))).sum() / x.sum() * 100 if x.sum() > 0 else 0)
    sec      = sec.join(top20.rename('Top20_Contribution_%'))
    sec['Is_Top_Heavy'] = sec['Top20_Contribution_%'] > 80

    # Monthly trends per section
    sec_monthly = df2.groupby(['Section', 'YM'], sort=False).agg(
        rev=('bal Value', 'sum'), qty=('bal Qty', 'sum')).reset_index()
    sec_trends = []
    for section, grp in tqdm(sec_monthly.groupby('Section', sort=False),
                              desc="Sections", ncols=80):
        mr, mq = grp['rev'].values, grp['qty'].values
        seas = _seasonal_strength(mr) if len(mr) >= 24 else 0.0
        is_s = seas > 0.3
        tail = lambda v: v[-12:] if len(v) > 12 else v
        cons = lambda v: (max((np.diff(tail(v)) > 0).sum(), (np.diff(tail(v)) < 0).sum()) / max(len(np.diff(tail(v))), 1) * 100)
        if len(mr) >= 6:
            rd, rc = _vote(_ema_trend(mr), *_spearman_trend(tail(mr)), _period_change(mr),
                           _yoy_change(mr) if is_s else None, _robust_cv(mr), cons(mr))
            qd, qc = _vote(_ema_trend(mq), *_spearman_trend(tail(mq)), _period_change(mq),
                           _yoy_change(mq) if is_s else None, _robust_cv(mq), cons(mq))
        else:
            rd = qd = 'Unknown'; rc = qc = 'LOW'
        mm = rd != qd and rc != 'LOW' and qc != 'LOW'
        sec_trends.append({'Section': section,
            'Revenue_Trend': rd, 'Revenue_Confidence': rc,
            'Revenue_Period_Change_%': float(_period_change(mr)),
            'Revenue_YoY_Change_%': float(_yoy_change(mr) or 0),
            'Quantity_Trend': qd, 'Quantity_Confidence': qc,
            'Quantity_Period_Change_%': float(_period_change(mq)),
            'Quantity_YoY_Change_%': float(_yoy_change(mq) or 0),
            'Is_Seasonal': is_s, 'Seasonal_Strength': float(seas),
            'Active_Months': len(mr),
            'Trend_Mismatch': mm,
            'Mismatch_Warning': ("Revenue UP / Qty DOWN" if rd == 'Growing' and qd == 'Declining'
                                 else "Revenue DOWN / Qty UP" if rd == 'Declining' and qd == 'Growing'
                                 else None) if mm else None})

    sec = sec.join(pd.DataFrame(sec_trends).set_index('Section'), how='left')

    # Section status
    m = sec.reset_index()
    rv = m.get('Revenue_Trend', pd.Series('Unknown', index=m.index))
    c_lm  = _b(m['Recent_Avg_Margin_%'] < 0)
    c_oos = ~c_lm & _b(m['Stock_Out_Rate_%'] > 30)
    c_in  = ~c_lm & ~c_oos & _b(m['Days_Since_Last_Sale'] > 180)
    c_dlm = _b(rv == 'Declining') & _b(m['Recent_Avg_Margin_%'] < 15)
    c_dec = _b(rv == 'Declining') & ~c_dlm
    c_th  = ~c_dlm & ~c_dec & _b(m['Is_Top_Heavy'])
    c_ghm = _b(rv == 'Growing') & _b(m['Recent_Avg_Margin_%'] > 30)
    c_gro = _b(rv == 'Growing') & ~c_ghm
    c_sea = _b(m.get('Is_Seasonal', pd.Series(False, index=m.index)))
    c_sp  = _b(rv == 'Stable') & _b(m['Recent_Avg_Margin_%'] > 25)
    conds = [c_lm, c_oos, c_in, c_dlm, c_dec, c_th, c_ghm, c_gro, c_sea, c_sp]
    slabels = ["Loss-Making", "High Stock-Out Rate", "Inactive",
               "Declining (Low Margin)", "Declining", "High Concentration Risk",
               "Growing (High Margin)", "Growing", "Seasonal", "Stable (Profitable)"]
    ma = m['Recent_Avg_Margin_%'].round(1).astype(str)
    m['Status'] = np.select(conds, slabels, default='Stable')
    m['Key_Reason'] = np.select(conds, [
        "Negative margin " + ma + "%",
        m['Stock_Out_Rate_%'].round(0).astype(int).astype(str) + "% SKUs out of stock",
        m['Days_Since_Last_Sale'].astype(str) + " days since last sale",
        "Revenue declining, " + ma + "% margin",
        "Revenue trending down",
        "Top 20% SKUs = " + m['Top20_Contribution_%'].round(0).astype(int).astype(str) + "% of revenue",
        "Revenue up, " + ma + "% margin",
        "Revenue trending up",
        "Clear seasonal pattern",
        "Steady, " + ma + "% margin",
    ], default="Consistent performance")

    return m.sort_values('Recent_Revenue', ascending=False)


# ── Excel export ──────────────────────────────────────────────────────────────

def _save_excel(sku_df: pd.DataFrame, section_df, path: str):
    print(f"\n── Saving → {path}")
    try:
        with pd.ExcelWriter(path, engine='openpyxl') as w:
            sku_df.to_excel(w, sheet_name='All SKUs', index=False)
            def ws(name, frame):
                if len(frame): frame.to_excel(w, sheet_name=name, index=False)
            ws('Critical SKUs',         sku_df[sku_df['Priority'] <= 5])
            ws('True Stockouts',        sku_df[sku_df['Is_True_Stockout']].sort_values('Estimated_Lost_Revenue', ascending=False))
            if 'Is_Potential_Stockout' in sku_df.columns:
                ws('Potential Stockouts', sku_df[sku_df['Is_Potential_Stockout']].sort_values('Estimated_Lost_Revenue', ascending=False))
            ws('Low Stock High Margin', sku_df[(sku_df['Current_Stock'] < 10) & (sku_df['Recent_Avg_Margin_%'] > 30)])
            if 'Trend_Mismatch' in sku_df.columns:
                ws('Trend Mismatches',   sku_df[sku_df['Trend_Mismatch']])
            ws('Growing SKUs',          sku_df[sku_df['Revenue_Trend'] == 'Growing'])
            ws('Declining SKUs',        sku_df[sku_df['Revenue_Trend'] == 'Declining'])
            if 'Is_Seasonal' in sku_df.columns:
                ws('Seasonal SKUs',     sku_df[sku_df['Is_Seasonal']])
            if section_df is not None and len(section_df):
                section_df.to_excel(w, sheet_name='All Sections', index=False)
                ws('Problem Sections', section_df[section_df['Status'].str.contains('Loss|Stockout|Declining|Risk', na=False)])
                if 'Is_Top_Heavy' in section_df.columns:
                    ws('Top Heavy Sections', section_df[section_df['Is_Top_Heavy']])
                if len(section_df) >= 20:
                    section_df.nlargest(20, 'Recent_Revenue').to_excel(w, sheet_name='Top 20 Sections', index=False)
        print(f"  saved  ·  {path}")
    except Exception as e:
        print(f"  save failed: {e}")


# ── Public API ────────────────────────────────────────────────────────────────

def ultimate_fabric_analysis(df: pd.DataFrame, analysis_date=None,
                              save_excel: bool = True,
                              output_file: str = "FABRIC_ANALYSIS_COMPLETE.xlsx"):
    """
    Full SKU + Section classification pipeline.

    Parameters
    ----------
    df            : DataFrame — requires SKU, Date, bal Value, bal Qty.
                    Optional: Section, CURRENT_STOCK, Total_Profit.
    analysis_date : Reference date (defaults to max date in df).
    save_excel    : Write Excel output.
    output_file   : Output path.

    Returns
    -------
    sku_df, section_df : DataFrames
    """
    df = df.copy()
    df['SKU_CLEAN'] = df['SKU'].astype(str).str.strip()
    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df['Date'] = pd.to_datetime(df['Date'])

    analysis_date = pd.Timestamp(analysis_date or df['Date'].max())
    has_profit    = 'Total_Profit' in df.columns

    print("\n── Classification Pipeline ───────────────────────────────────")
    print(f"  analysis date : {analysis_date.date()}")
    print(f"  unique SKUs   : {df['SKU_CLEAN'].nunique():,}")
    print(f"  transactions  : {len(df):,}")
    print(f"  profit data   : {'yes' if has_profit else 'no'}")
    print(f"  stock data    : {'yes' if 'CURRENT_STOCK' in df.columns else 'no'}")

    # Pre-compute section seasonality (STL once per section, not per SKU)
    sec_seasonal: dict = {}
    if 'Section' in df.columns:
        print("\n── Pre-computing section seasonality ─────────────────────────")
        df['_ym'] = df['Date'].dt.to_period('M')
        for sec, grp in tqdm(df.groupby(['Section', '_ym'])['bal Value'].sum()
                              .groupby('Section'), desc="STL", ncols=80):
            sec_seasonal[str(sec)] = _seasonal_strength(grp.values)
        df.drop(columns=['_ym'], inplace=True)
        gc.collect()

    print("\n── Stage 1 · Scalar metrics (vectorised) ─────────────────────")
    scalar = _scalar_metrics(df, analysis_date)
    print(f"  {len(scalar):,} SKUs")
    gc.collect()

    print("\n── Stage 2 · Monthly trend signals ──────────────────────────")
    trends = _monthly_trends(df, sec_seasonal, analysis_date)
    print(f"  {len(trends):,} SKUs")
    gc.collect()

    sku_df = scalar.merge(trends, on='SKU_CLEAN', how='left')

    print("\n── Stage 3 · Lost-sales estimation ──────────────────────────")
    sku_df = _lost_sales(df, sku_df, analysis_date, has_profit)
    print(f"  {sku_df['Is_Stockout'].sum():,} stockout SKUs processed")
    gc.collect()

    print("\n── Stage 4 · Status assignment (vectorised) ─────────────────")
    # Rename for _assign_sku_status compatibility
    sku_df = sku_df.rename(columns={
        'Total_Txns': 'Total_Transactions',
        'Recent_Txns': 'Recent_Txns',  # kept as-is for condition
        'Days_Since_Last': 'Days_Since_Last',
    })
    sku_df = _assign_sku_status(sku_df)
    sku_df = sku_df.sort_values(['Priority', 'Recent_Revenue'], ascending=[True, False])
    print(f"  {len(sku_df):,} SKUs classified")

    print("\n── Stage 5 · Section analysis ───────────────────────────────")
    section_df = _section_analysis(df, sku_df, analysis_date) if 'Section' in df.columns else None
    if section_df is not None:
        print(f"  {len(section_df):,} sections")

    # Clean up column names for output
    sku_df.rename(columns={'SKU_CLEAN': 'SKU'}, inplace=True, errors='ignore')

    if save_excel:
        _save_excel(sku_df, section_df, output_file)

    return sku_df, section_df
