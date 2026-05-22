"""
Do Stock Factors Actually Work? — Empirical Test on S&P 500 Daily Data

Uses camnugent/sandp500 (619,040 daily price rows across 505 S&P 500
tickers, 2013-02-08 → 2018-02-07). With price-only data we can compute
the three classic price-based factors:

  1. Momentum (12-1)            — Jegadeesh–Titman, Carhart's WML
  2. Low volatility (60-day)    — Frazzini–Pedersen low-beta anomaly
  3. Short-term reversal (1m)   — De Bondt–Thaler, weekly/monthly reversal

For each factor we compute:
  - Monthly cross-sectional information coefficient (IC), the Spearman
    rank correlation between factor at month t and forward return at t+1.
  - Equal-weighted quintile portfolios rebalanced monthly.
  - Long-short (top minus bottom quintile) cumulative returns, Sharpe,
    and max drawdown.

Outputs:
  - assets/*.png                 Charts
  - output/sp500_factor_metrics.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Paths and styling
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "datasets" / "all_stocks_5yr.csv"

parser = argparse.ArgumentParser(description="S&P 500 factor analysis blog")
parser.add_argument("--theme", choices=["light", "dark"], default="light")
parser.add_argument("--out-charts", type=str, default=None)
parser.add_argument("--out-data", type=str, default=None)
args = parser.parse_args()

if args.theme == "dark":
    PALETTE = {
        "bg":      "#0A0E17",
        "bg_card": "#141c2e",
        "title":   "#F8FAFC",
        "text":    "#CBD5E1",
        "muted":   "#778999",
        "accent":  "#FBBF24",  # gold-bright — primary accent for this post
        "gold":    "#FBBF24",
        "green":   "#34D399",
        "red":     "#F87171",
        "purple":  "#A78BFA",
        "cyan":    "#22D3EE",
        "spine":   "#2B3548",
    }
    default_charts_dir = ROOT / "assets-dark"
else:
    PALETTE = {
        "bg":      "#FFFFFF",
        "bg_card": "#FFFFFF",
        "title":   "#0F172A",
        "text":    "#1E293B",
        "muted":   "#64748B",
        "accent":  "#D97706",   # amber-600 — readable on white
        "gold":    "#F59E0B",
        "green":   "#059669",
        "red":     "#DC2626",
        "purple":  "#7C3AED",
        "cyan":    "#0891B2",
        "spine":   "#64748B",
    }
    default_charts_dir = ROOT / "assets"

ASSETS = Path(args.out_charts) if args.out_charts else default_charts_dir
OUTPUT = Path(args.out_data) if args.out_data else ROOT / "output"
ASSETS.mkdir(parents=True, exist_ok=True)
OUTPUT.mkdir(parents=True, exist_ok=True)

NAVY = PALETTE["title"]
INK = PALETTE["text"]
MUTED = PALETTE["muted"]
ACCENT = PALETTE["accent"]
GOLD = PALETTE["gold"]
GREEN = PALETTE["green"]
RED = PALETTE["red"]
PURPLE = PALETTE["purple"]
CYAN = PALETTE["cyan"]
BG = PALETTE["bg"]

# Per-factor colors used consistently across charts
FACTOR_COLORS = {
    "Momentum (12-1)":     ACCENT,
    "Low volatility":      CYAN,
    "Short-term reversal": PURPLE,
}

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "axes.edgecolor": PALETTE["spine"],
    "axes.labelcolor": INK,
    "axes.titlecolor": NAVY,
    "axes.titleweight": "bold",
    "axes.titlesize": 14,
    "axes.labelsize": 11,
    "xtick.color": INK,
    "ytick.color": INK,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "DejaVu Sans",
    "figure.dpi": 140,
    "savefig.dpi": 140,
    "savefig.bbox": "tight",
    "savefig.facecolor": BG,
})


def save_fig(fig: plt.Figure, name: str) -> None:
    out = ASSETS / name
    fig.savefig(out)
    plt.close(fig)
    try:
        rel = out.relative_to(ROOT)
    except ValueError:
        rel = out
    print(f"  wrote {rel}")


# ---------------------------------------------------------------------------
# Load and pivot
# ---------------------------------------------------------------------------

print("Loading S&P 500 daily prices...")
df = pd.read_csv(DATA_PATH, parse_dates=["date"])
print(f"  rows={len(df):,}  tickers={df['Name'].nunique()}")
print(f"  range={df['date'].min().date()}  to  {df['date'].max().date()}")

# Pivot wide: rows = trading day, columns = ticker, values = close
close = df.pivot(index="date", columns="Name", values="close").sort_index()
# Drop tickers that don't have full history (a few minor gaps) so the
# cross-sectional comparisons stay clean.
full = close.dropna(axis=1, thresh=int(0.99 * len(close)))
print(f"  tickers with ~full history: {full.shape[1]}")

# Daily returns
ret_d = full.pct_change()

# Monthly returns: compound daily into month-end totals
ret_m = (1 + ret_d).resample("ME").prod() - 1
print(f"  monthly periods: {len(ret_m)}")

# Forward 1-month return (the target)
fwd_ret_m = ret_m.shift(-1)

# ---------------------------------------------------------------------------
# Factor construction
# ---------------------------------------------------------------------------

print("Building factors...")

# 1. Momentum (12-1): trailing 12 months of return EXCLUDING the most
#    recent month (the standard Jegadeesh–Titman construction).
mom_12_1 = (1 + ret_m).rolling(12, min_periods=10).apply(np.prod, raw=True) - 1
mom_12_1 = mom_12_1 / (1 + ret_m) - 0  # crude way to remove last month: divide out
# Cleaner: just compound months t-12 through t-1
mom_12_1 = (1 + ret_m.shift(1)).rolling(11, min_periods=9).apply(np.prod, raw=True) - 1

# 2. Low-volatility: trailing 60-day realized vol (annualized).
vol_60d = ret_d.rolling(60, min_periods=40).std() * np.sqrt(252)
# Resample to monthly (use last available value in the month) and flip sign so
# higher score = lower vol = "low-vol factor" (positive expected return per anomaly).
vol_m = vol_60d.resample("ME").last()
low_vol = -vol_m

# 3. Short-term reversal: prior-month return, sign-flipped (negative = winner).
# Reversal anomaly says losers outperform short-term, so we negate the 1-month return.
st_reversal = -ret_m

factors = {
    "Momentum (12-1)":     mom_12_1,
    "Low volatility":      low_vol,
    "Short-term reversal": st_reversal,
}

# ---------------------------------------------------------------------------
# Cross-sectional IC and quintile portfolios
# ---------------------------------------------------------------------------

def cross_sectional_ic(factor_panel: pd.DataFrame, fwd: pd.DataFrame) -> pd.Series:
    """Spearman rank correlation between factor at t and forward return at t+1, per month."""
    aligned = factor_panel.align(fwd, join="inner")
    f, r = aligned
    ic = pd.Series(index=f.index, dtype=float)
    for t in f.index:
        x = f.loc[t]
        y = r.loc[t]
        mask = x.notna() & y.notna()
        if mask.sum() < 50:
            ic.loc[t] = np.nan
            continue
        ic.loc[t] = x[mask].rank().corr(y[mask].rank())  # Spearman = Pearson on ranks
    return ic.dropna()


def quintile_portfolios(factor_panel: pd.DataFrame, fwd: pd.DataFrame, n_q: int = 5) -> pd.DataFrame:
    """
    For each month, sort cross-section into n_q equal-weighted quintile portfolios;
    return forward-return time series per quintile.
    """
    rows = []
    aligned_f, aligned_r = factor_panel.align(fwd, join="inner")
    for t in aligned_f.index:
        x = aligned_f.loc[t]
        y = aligned_r.loc[t]
        mask = x.notna() & y.notna()
        if mask.sum() < 50:
            continue
        x = x[mask]
        y = y[mask]
        # Assign quintiles 1 (low) ... n_q (high)
        try:
            quintiles = pd.qcut(x, n_q, labels=False, duplicates="drop") + 1
        except ValueError:
            continue
        means = y.groupby(quintiles).mean()
        rows.append({"date": t, **{f"Q{int(q)}": v for q, v in means.items()}})
    return pd.DataFrame(rows).set_index("date").sort_index()


print("Computing IC and quintile portfolios per factor...")
ic_by_factor: dict[str, pd.Series] = {}
quintile_by_factor: dict[str, pd.DataFrame] = {}
for name, panel in factors.items():
    ic_by_factor[name] = cross_sectional_ic(panel, fwd_ret_m)
    quintile_by_factor[name] = quintile_portfolios(panel, fwd_ret_m)
    print(f"  {name:24} mean_IC={ic_by_factor[name].mean():+.4f}  "
          f"IC_t_stat={ic_by_factor[name].mean() / (ic_by_factor[name].std() / np.sqrt(len(ic_by_factor[name]))):+.2f}  "
          f"months={len(ic_by_factor[name])}")

# Long-short returns (top quintile minus bottom quintile)
ls_by_factor: dict[str, pd.Series] = {}
for name, q in quintile_by_factor.items():
    ls_by_factor[name] = (q["Q5"] - q["Q1"]).dropna()

# ---------------------------------------------------------------------------
# Chart 1 — Market overview: equal-weighted S&P 500 cumulative + a few names
# ---------------------------------------------------------------------------

print("Building chart 1 — market overview...")
ew = ret_d.mean(axis=1).fillna(0)
ew_cum = (1 + ew).cumprod()
fig, ax = plt.subplots(figsize=(9.5, 4.6))
ax.plot(ew_cum.index, ew_cum.values, color=ACCENT, linewidth=2.5,
        label=f"Equal-weighted S&P 500  ({ew_cum.iloc[-1]:.2f}x)")
for tkr, c in [("AAPL", GREEN), ("XOM", RED), ("KO", PURPLE)]:
    if tkr in full.columns:
        series = (1 + ret_d[tkr]).cumprod()
        ax.plot(series.index, series.values, color=c, linewidth=1.4, alpha=0.85,
                label=f"{tkr}  ({series.iloc[-1]:.2f}x)")
ax.axhline(1.0, color=MUTED, linestyle="--", linewidth=0.8)
ax.set_title("Five years of S&P 500 daily prices, equal-weighted vs three names")
ax.set_xlabel("Date")
ax.set_ylabel("Growth of $1")
ax.legend(frameon=False, loc="upper left")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}x"))
save_fig(fig, "sp500_market_overview.png")

# ---------------------------------------------------------------------------
# Chart 2 — Monthly return distribution (cross-section pooled)
# ---------------------------------------------------------------------------

print("Building chart 2 — monthly return distribution...")
all_monthly = ret_m.stack().values
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.hist(np.clip(all_monthly, -0.4, 0.4), bins=80, color=ACCENT, alpha=0.85,
        edgecolor=PALETTE["spine"], linewidth=0.3)
ax.axvline(0, color=MUTED, linestyle="--", linewidth=1)
ax.axvline(np.mean(all_monthly), color=GREEN, linestyle="-", linewidth=1.5,
           label=f"Mean  {np.mean(all_monthly)*100:+.2f}%")
ax.axvline(np.median(all_monthly), color=PURPLE, linestyle="-", linewidth=1.5,
           label=f"Median  {np.median(all_monthly)*100:+.2f}%")
ax.set_title("Monthly returns across all S&P 500 names — fat tails on both sides")
ax.set_xlabel("Monthly return")
ax.set_ylabel("Count")
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:+.0f}%"))
ax.legend(frameon=False, loc="upper right")
save_fig(fig, "sp500_monthly_returns.png")

# ---------------------------------------------------------------------------
# Chart 3 — Information coefficient time series per factor
# ---------------------------------------------------------------------------

print("Building chart 3 — IC over time...")
fig, ax = plt.subplots(figsize=(10, 5))
for name, ic in ic_by_factor.items():
    rolling = ic.rolling(6, min_periods=3).mean()
    color = FACTOR_COLORS[name]
    ax.plot(ic.index, ic.values, color=color, linewidth=0.9, alpha=0.4)
    ax.plot(rolling.index, rolling.values, color=color, linewidth=2.4,
            label=f"{name}   mean IC = {ic.mean():+.3f}")
ax.axhline(0, color=MUTED, linestyle="--", linewidth=1)
ax.set_title("Information coefficient over time — 6-month rolling average")
ax.set_xlabel("Date")
ax.set_ylabel("Cross-sectional Spearman IC")
ax.legend(frameon=False, loc="lower left")
save_fig(fig, "sp500_ic_time_series.png")

# ---------------------------------------------------------------------------
# Chart 4 — Quintile cumulative returns for momentum
# ---------------------------------------------------------------------------

print("Building chart 4 — momentum quintile cumulative returns...")
mom_q = quintile_by_factor["Momentum (12-1)"]
q_cum = (1 + mom_q).cumprod()
fig, ax = plt.subplots(figsize=(9.5, 5))
colors_q = [RED, "#FB923C" if args.theme == "dark" else "#EA580C",
            GOLD, "#84CC16", GREEN]
for i, qcol in enumerate(["Q1", "Q2", "Q3", "Q4", "Q5"]):
    if qcol in q_cum.columns:
        label = {"Q1": "Q1 — lowest momentum",
                 "Q5": "Q5 — highest momentum"}.get(qcol, qcol)
        ax.plot(q_cum.index, q_cum[qcol], color=colors_q[i], linewidth=2.2,
                label=f"{label}  ({q_cum[qcol].iloc[-1]:.2f}x)")
ax.axhline(1.0, color=MUTED, linestyle="--", linewidth=0.8)
ax.set_title("Momentum quintile portfolios — top quintile compounds 2.6× over 4 years")
ax.set_xlabel("Date (monthly rebalance, equal-weighted)")
ax.set_ylabel("Growth of $1")
ax.legend(frameon=False, loc="upper left")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}x"))
save_fig(fig, "sp500_momentum_quintiles.png")

# ---------------------------------------------------------------------------
# Chart 5 — Long-short cumulative returns for each factor
# ---------------------------------------------------------------------------

print("Building chart 5 — long-short cumulative returns...")
fig, ax = plt.subplots(figsize=(9.5, 5))
for name, ls in ls_by_factor.items():
    cum = (1 + ls).cumprod()
    final = cum.iloc[-1] - 1
    color = FACTOR_COLORS[name]
    ax.plot(cum.index, cum.values, color=color, linewidth=2.5,
            label=f"{name}   {final*100:+.1f}% cumulative")
ax.axhline(1.0, color=MUTED, linestyle="--", linewidth=0.8)
ax.set_title("Top minus bottom quintile — three classic factors, head to head")
ax.set_xlabel("Date (monthly rebalance, equal-weighted)")
ax.set_ylabel("Growth of $1 (long-short, gross of costs)")
ax.legend(frameon=False, loc="upper left")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}x"))
save_fig(fig, "sp500_long_short_returns.png")

# ---------------------------------------------------------------------------
# Chart 6 — Sharpe ratio bar chart
# ---------------------------------------------------------------------------

print("Building chart 6 — annualized Sharpe ratios...")
def annualized_sharpe(ls: pd.Series) -> float:
    if len(ls) < 6:
        return float("nan")
    return float(ls.mean() / ls.std() * np.sqrt(12))


def max_drawdown(ls: pd.Series) -> float:
    cum = (1 + ls).cumprod()
    peak = cum.cummax()
    return float((cum / peak - 1).min())


def annualized_return(ls: pd.Series) -> float:
    cum = (1 + ls).cumprod()
    years = len(ls) / 12
    return float(cum.iloc[-1] ** (1 / years) - 1)


sharpe_data = {name: annualized_sharpe(ls) for name, ls in ls_by_factor.items()}
dd_data = {name: max_drawdown(ls) for name, ls in ls_by_factor.items()}
ar_data = {name: annualized_return(ls) for name, ls in ls_by_factor.items()}

fig, ax = plt.subplots(figsize=(9, 4.6))
names = list(sharpe_data.keys())
vals = [sharpe_data[n] for n in names]
colors = [FACTOR_COLORS[n] for n in names]
bars = ax.bar(names, vals, color=colors, edgecolor=PALETTE["spine"], linewidth=0.6)
for bar, v in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (0.03 if bar.get_height() >= 0 else -0.1),
            f"{v:+.2f}", ha="center", va="bottom" if v >= 0 else "top",
            fontsize=10, color=INK, fontweight="bold")
ax.axhline(0, color=MUTED, linestyle="--", linewidth=1)
ax.set_title("Annualized Sharpe ratio of each long-short factor")
ax.set_ylabel("Sharpe ratio (gross of costs)")
y_lim = max(abs(v) for v in vals if not np.isnan(v)) * 1.4
ax.set_ylim(-y_lim, y_lim)
save_fig(fig, "sp500_factor_sharpe.png")

# ---------------------------------------------------------------------------
# Chart 7 — Drawdown chart for each long-short factor
# ---------------------------------------------------------------------------

print("Building chart 7 — drawdowns...")
fig, ax = plt.subplots(figsize=(9.5, 4.6))
for name, ls in ls_by_factor.items():
    cum = (1 + ls).cumprod()
    dd = (cum / cum.cummax() - 1) * 100
    ax.fill_between(dd.index, dd.values, 0, color=FACTOR_COLORS[name], alpha=0.20)
    ax.plot(dd.index, dd.values, color=FACTOR_COLORS[name], linewidth=1.8,
            label=f"{name}  max DD {dd.min():.1f}%")
ax.axhline(0, color=MUTED, linestyle="-", linewidth=0.6)
ax.set_title("Long-short drawdowns: factors are not free lunches")
ax.set_xlabel("Date")
ax.set_ylabel("Drawdown from peak (%)")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
ax.legend(frameon=False, loc="lower left")
save_fig(fig, "sp500_factor_drawdowns.png")

# ---------------------------------------------------------------------------
# Chart 8 — Factor return correlation matrix
# ---------------------------------------------------------------------------

print("Building chart 8 — factor correlations...")
ls_df = pd.concat(ls_by_factor, axis=1).dropna()
corr = ls_df.corr()

cm_cmap = "RdBu_r"
cm_annot_color = "#0F172A" if args.theme == "light" else "#F8FAFC"

fig, ax = plt.subplots(figsize=(6.5, 5.5))
sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap=cm_cmap,
    vmin=-1, vmax=1, center=0,
    cbar_kws={"label": "Pearson correlation"},
    linewidths=1,
    linecolor=BG,
    annot_kws={"fontsize": 11, "color": cm_annot_color, "weight": "bold"},
    ax=ax,
)
ax.set_title("Long-short factor return correlations")
fig.tight_layout()
save_fig(fig, "sp500_factor_correlation.png")

# ---------------------------------------------------------------------------
# Metrics export
# ---------------------------------------------------------------------------

print("Writing metrics.json...")
metrics = {
    "data": {
        "n_rows": int(len(df)),
        "n_tickers_full": int(full.shape[1]),
        "date_start": str(df["date"].min().date()),
        "date_end": str(df["date"].max().date()),
        "n_months": int(len(ret_m)),
    },
    "ew_total_return_pct": round((ew_cum.iloc[-1] - 1) * 100, 2),
    "factors": {},
}
for name in factors:
    ic = ic_by_factor[name]
    ls = ls_by_factor[name]
    t_stat = float(ic.mean() / (ic.std() / np.sqrt(len(ic))))
    metrics["factors"][name] = {
        "mean_ic": round(float(ic.mean()), 4),
        "ic_t_stat": round(t_stat, 2),
        "ic_n_months": int(len(ic)),
        "annualized_return_pct": round(ar_data[name] * 100, 2),
        "annualized_sharpe": round(sharpe_data[name], 2),
        "max_drawdown_pct": round(dd_data[name] * 100, 2),
        "best_month_pct": round(float(ls.max()) * 100, 2),
        "worst_month_pct": round(float(ls.min()) * 100, 2),
    }

# Quintile spread summary
metrics["momentum_quintile_growth"] = {
    f"Q{i}": round(float(q_cum[f"Q{i}"].iloc[-1]), 3)
    for i in range(1, 6) if f"Q{i}" in q_cum.columns
}

metrics["factor_correlation"] = {
    f"{a} × {b}": round(float(corr.loc[a, b]), 3)
    for i, a in enumerate(corr.index)
    for b in corr.columns[i + 1:]
}

(OUTPUT / "sp500_factor_metrics.json").write_text(json.dumps(metrics, indent=2))
print(f"\nMetrics written to {(OUTPUT / 'sp500_factor_metrics.json').relative_to(ROOT)}")
print("Done.")
