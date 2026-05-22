"""
Do Stock Factors Work in Crypto? — Empirical Test on 1,000+ Tokens

Uses ayushkhaire/top-1000-cryptos-historical (8.4M rows, ~8,500 unique
tickers across 2014–2026). After liquidity filtering (mean daily dollar
volume between $1M and $50B) and history requirements (>=500 days from
2020 onwards), we keep ~1,067 cryptos for cross-sectional factor
analysis.

This is the crypto analog of analysis/sp500_factor_test.py — same three
price-based factors, same cross-sectional IC + quintile portfolio
methodology, so results compare cleanly to the S&P 500 post.

  1. Momentum (12-1)            — trailing 12 months excl. last month
  2. Low volatility (60-day)    — sign-flipped realized vol
  3. Short-term reversal (1m)   — sign-flipped prior-month return

Crypto-specific considerations:
  - Winsorize daily returns at the 99.5th percentile (crypto has fat
    tails that would dominate cross-sectional stats otherwise).
  - Drop stablecoins (USDT, USDC, DAI, etc.) — pegged assets have ~0
    return and break cross-sectional ranking.
  - Survivorship bias acknowledged: the dataset only contains tokens
    that exist as of compilation. Dead coins from the 2021/2022 bear
    market that fully zeroed out are not represented.

Outputs:
  - assets/crypto_*.png
  - output/crypto_factor_metrics.json
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Paths and styling
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "datasets"

parser = argparse.ArgumentParser(description="Crypto factor analysis blog")
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
        "accent":  "#22D3EE",      # cyan — distinguishes from previous post's gold
        "gold":    "#FBBF24",
        "green":   "#34D399",
        "red":     "#F87171",
        "purple":  "#A78BFA",
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
        "accent":  "#0891B2",
        "gold":    "#F59E0B",
        "green":   "#059669",
        "red":     "#DC2626",
        "purple":  "#7C3AED",
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
BG = PALETTE["bg"]

FACTOR_COLORS = {
    "Momentum (12-1)":     ACCENT,
    "Low volatility":      GOLD,
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
# Load + filter
# ---------------------------------------------------------------------------

# Stablecoin tickers to drop — pegged assets break cross-sectional ranking.
STABLE_NAMES = {
    "USDT", "USDC", "DAI", "BUSD", "TUSD", "USDP", "USDD", "UST", "FRAX",
    "LUSD", "FDUSD", "PYUSD", "GUSD", "EURT", "EURS", "MIM", "USDE",
    "USD0", "USDV", "USDX", "USDS", "USDC0", "USDC1", "USDX1", "USDS1",
}


def is_stable(ticker: str) -> bool:
    base = re.split(r"[-_\d]", ticker, maxsplit=1)[0]
    return base in STABLE_NAMES


print("Loading and concatenating crypto data files...")
files = sorted(glob.glob(str(DATA_DIR / "[0-9A-Z].csv")))
dfs = []
for f in files:
    try:
        d = pd.read_csv(f)
        if len(d) > 0:
            dfs.append(d)
    except Exception:
        pass
raw = pd.concat(dfs, ignore_index=True)
raw["date"] = pd.to_datetime(raw["timestamp"], unit="s")
print(f"  raw rows={len(raw):,}  tickers={raw['ticker'].nunique():,}")

# Restrict to 2020 onwards — the pre-2020 crypto market was tiny and
# unrepresentative of how the asset class actually trades today.
df = raw[raw["date"] >= "2020-01-01"].copy()
df = df.sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date"])

# Drop stablecoins
before = df["ticker"].nunique()
df = df[~df["ticker"].apply(is_stable)]
print(f"  dropped {before - df['ticker'].nunique()} stablecoin tickers")

# Liquidity filter: mean daily $ volume between $1M and $50B (the upper
# bound rejects obviously-bad data — no real token has trillions in daily
# volume; the lower bound drops dust-tier tokens with no real liquidity).
dollar_vol = (df["volume"] * df["close"]).groupby(df["ticker"]).mean()
liquid_set = dollar_vol[(dollar_vol >= 1_000_000) & (dollar_vol <= 50_000_000_000)].index
df = df[df["ticker"].isin(liquid_set)]

# History requirement: at least 500 trading days (needed for 12-month
# momentum lookback with a comfortable buffer).
days_per = df.groupby("ticker").size()
ok_tickers = days_per[days_per >= 500].index
df = df[df["ticker"].isin(ok_tickers)].copy()

n_tickers = df["ticker"].nunique()
date_start = df["date"].min().date()
date_end = df["date"].max().date()
print(f"  after filtering: rows={len(df):,}  tickers={n_tickers}")
print(f"  date range: {date_start} to {date_end}")

# Pivot wide
close = df.pivot(index="date", columns="ticker", values="close").sort_index()
# Daily returns — winsorize at the 99.5th percentile each tail (crypto
# fat tails would otherwise dominate cross-sectional statistics).
ret_d = close.pct_change()
WINS = ret_d.stack().quantile([0.005, 0.995]).values
ret_d = ret_d.clip(lower=WINS[0], upper=WINS[1])
print(f"  daily-return winsorization: [{WINS[0]:.4f}, {WINS[1]:.4f}]")

# Monthly returns
ret_m = (1 + ret_d).resample("ME").prod() - 1

# Cross-sectional winsorize per month at 5/95 percentiles. The daily
# winsorize already trims single-day extremes; this trims the (much
# more impactful) single-MONTH extremes where a microcap pumps 50× and
# would otherwise dominate the equal-weighted quintile portfolio mean.
# Rank-based metrics (IC) are unaffected; the change shows up in the
# long-short magnitudes, which become defensible operational numbers
# instead of "203 million percent annualized."
ret_m_wins = ret_m.copy()
for t in ret_m_wins.index:
    row = ret_m_wins.loc[t]
    if row.notna().sum() < 100:
        continue
    lo, hi = row.quantile([0.05, 0.95])
    ret_m_wins.loc[t] = row.clip(lo, hi)
fwd_ret_m = ret_m_wins.shift(-1)
print(f"  monthly periods: {len(ret_m)}  (winsorized cross-sectionally at 5/95 per month)")

# ---------------------------------------------------------------------------
# Factors (identical construction to sp500_factor_test.py)
# ---------------------------------------------------------------------------

print("Building factors...")
mom_12_1 = (1 + ret_m.shift(1)).rolling(11, min_periods=9).apply(np.prod, raw=True) - 1
vol_60d = ret_d.rolling(60, min_periods=40).std() * np.sqrt(365)  # 365 for crypto, 24/7
low_vol = -vol_60d.resample("ME").last()
st_reversal = -ret_m

factors = {
    "Momentum (12-1)":     mom_12_1,
    "Low volatility":      low_vol,
    "Short-term reversal": st_reversal,
}


def cross_sectional_ic(panel: pd.DataFrame, fwd: pd.DataFrame) -> pd.Series:
    aligned_f, aligned_r = panel.align(fwd, join="inner")
    ic = pd.Series(index=aligned_f.index, dtype=float)
    for t in aligned_f.index:
        x = aligned_f.loc[t]
        y = aligned_r.loc[t]
        mask = x.notna() & y.notna()
        if mask.sum() < 100:
            ic.loc[t] = np.nan
            continue
        ic.loc[t] = x[mask].rank().corr(y[mask].rank())
    return ic.dropna()


def quintile_portfolios(panel: pd.DataFrame, fwd: pd.DataFrame, n_q: int = 5) -> pd.DataFrame:
    rows = []
    aligned_f, aligned_r = panel.align(fwd, join="inner")
    for t in aligned_f.index:
        x = aligned_f.loc[t]
        y = aligned_r.loc[t]
        mask = x.notna() & y.notna()
        if mask.sum() < 100:
            continue
        x = x[mask]
        y = y[mask]
        try:
            q = pd.qcut(x, n_q, labels=False, duplicates="drop") + 1
        except ValueError:
            continue
        means = y.groupby(q).mean()
        rows.append({"date": t, **{f"Q{int(k)}": v for k, v in means.items()}})
    return pd.DataFrame(rows).set_index("date").sort_index()


print("Computing IC and quintile portfolios per factor...")
ic_by_factor: dict[str, pd.Series] = {}
quint_by_factor: dict[str, pd.DataFrame] = {}
for name, panel in factors.items():
    ic = cross_sectional_ic(panel, fwd_ret_m)
    quint = quintile_portfolios(panel, fwd_ret_m)
    ic_by_factor[name] = ic
    quint_by_factor[name] = quint
    t_stat = ic.mean() / (ic.std() / np.sqrt(len(ic))) if len(ic) > 1 else float("nan")
    print(f"  {name:24} mean_IC={ic.mean():+.4f}  t_stat={t_stat:+.2f}  n={len(ic)}")

# Long-short returns
ls_by_factor: dict[str, pd.Series] = {
    name: (q["Q5"] - q["Q1"]).dropna() for name, q in quint_by_factor.items()
}

# ---------------------------------------------------------------------------
# Chart 1 — Universe overview: market-cap-weighted-ish (use BTC, ETH + agg)
# ---------------------------------------------------------------------------

print("Building chart 1 — universe overview...")
fig, ax = plt.subplots(figsize=(9.5, 4.6))

# Equal-weighted "crypto market" return over the universe
ew = ret_d.mean(axis=1).fillna(0)
ew_cum = (1 + ew).cumprod()
ax.plot(ew_cum.index, ew_cum.values, color=ACCENT, linewidth=2.5,
        label=f"Equal-weighted universe  ({ew_cum.iloc[-1]:.1f}x)")
for tkr, c, lbl in [("BTC-USD", GOLD, "BTC"), ("ETH-USD", PURPLE, "ETH")]:
    if tkr in close.columns:
        s = (1 + ret_d[tkr]).cumprod()
        ax.plot(s.index, s.values, color=c, linewidth=1.6, alpha=0.85,
                label=f"{lbl}  ({s.iloc[-1]:.1f}x)")
ax.axhline(1.0, color=MUTED, linestyle="--", linewidth=0.8)
ax.set_yscale("log")
ax.set_title("Crypto universe vs BTC/ETH, 2020 → today (log scale)")
ax.set_xlabel("Date")
ax.set_ylabel("Growth of $1 (log)")
ax.legend(frameon=False, loc="upper left")
save_fig(fig, "crypto_universe_overview.png")

# ---------------------------------------------------------------------------
# Chart 2 — Monthly return distribution (cross-section pooled)
# ---------------------------------------------------------------------------

print("Building chart 2 — monthly return distribution...")
pooled = ret_m.stack()
pooled = pooled[pooled.between(-0.95, 5)]  # show -95% to +500%
fig, ax = plt.subplots(figsize=(9.5, 4.6))
ax.hist(np.clip(pooled.values, -1, 2), bins=80, color=ACCENT, alpha=0.85,
        edgecolor=PALETTE["spine"], linewidth=0.3)
ax.axvline(0, color=MUTED, linestyle="--", linewidth=1)
ax.axvline(float(pooled.mean()), color=GREEN, linestyle="-", linewidth=1.5,
           label=f"Mean  {pooled.mean()*100:+.2f}%")
ax.axvline(float(pooled.median()), color=PURPLE, linestyle="-", linewidth=1.5,
           label=f"Median  {pooled.median()*100:+.2f}%")
ax.set_title("Crypto monthly returns: median is negative, mean positive — extreme right tail")
ax.set_xlabel("Monthly return (capped at ±100% for display)")
ax.set_ylabel("Count")
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:+.0f}%"))
ax.legend(frameon=False, loc="upper right")
save_fig(fig, "crypto_monthly_returns.png")

# ---------------------------------------------------------------------------
# Chart 3 — IC time series
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
ax.set_title("Information coefficient over time — 6-month rolling")
ax.set_xlabel("Date")
ax.set_ylabel("Cross-sectional Spearman IC")
ax.legend(frameon=False, loc="lower left")
save_fig(fig, "crypto_ic_time_series.png")

# ---------------------------------------------------------------------------
# Chart 4 — Momentum quintile cumulative returns
# ---------------------------------------------------------------------------

print("Building chart 4 — momentum quintile cumulative returns...")
mq = quint_by_factor["Momentum (12-1)"]
qc = (1 + mq).cumprod()
fig, ax = plt.subplots(figsize=(9.5, 5))
qcolors = [RED, "#FB923C" if args.theme == "dark" else "#EA580C",
           GOLD, "#84CC16", GREEN]
for i, qcol in enumerate(["Q1", "Q2", "Q3", "Q4", "Q5"]):
    if qcol in qc.columns:
        lbl = {"Q1": "Q1 — lowest momentum",
               "Q5": "Q5 — highest momentum"}.get(qcol, qcol)
        ax.plot(qc.index, qc[qcol], color=qcolors[i], linewidth=2.2,
                label=f"{lbl}  ({qc[qcol].iloc[-1]:.1f}x)")
ax.axhline(1.0, color=MUTED, linestyle="--", linewidth=0.8)
ax.set_yscale("log")
ax.set_title("Momentum quintiles — top vs bottom spread is wider in crypto than equities")
ax.set_xlabel("Date (monthly rebalance, equal-weighted)")
ax.set_ylabel("Growth of $1 (log)")
ax.legend(frameon=False, loc="upper left")
save_fig(fig, "crypto_momentum_quintiles.png")

# ---------------------------------------------------------------------------
# Chart 5 — Long-short cumulative
# ---------------------------------------------------------------------------

print("Building chart 5 — long-short cumulative returns...")
fig, ax = plt.subplots(figsize=(9.5, 5))
for name, ls in ls_by_factor.items():
    cum = (1 + ls).cumprod()
    final = cum.iloc[-1] - 1
    ax.plot(cum.index, cum.values, color=FACTOR_COLORS[name], linewidth=2.5,
            label=f"{name}   {final*100:+.0f}% cumulative")
ax.axhline(1.0, color=MUTED, linestyle="--", linewidth=0.8)
ax.set_title("Top minus bottom quintile — three factors, on crypto this time")
ax.set_xlabel("Date (monthly rebalance, equal-weighted)")
ax.set_ylabel("Growth of $1 (long-short, gross of costs)")
ax.legend(frameon=False, loc="upper left")
save_fig(fig, "crypto_long_short_returns.png")

# ---------------------------------------------------------------------------
# Chart 6 — Sharpe + return summary
# ---------------------------------------------------------------------------

print("Building chart 6 — Sharpe + return summary...")


def annualized(ls: pd.Series):
    if len(ls) < 6:
        return float("nan"), float("nan"), float("nan")
    cum = (1 + ls).cumprod()
    years = len(ls) / 12
    ann_ret = cum.iloc[-1] ** (1 / years) - 1
    sharpe = ls.mean() / ls.std() * np.sqrt(12)
    mdd = (cum / cum.cummax() - 1).min()
    return float(ann_ret), float(sharpe), float(mdd)


stats = {name: annualized(ls) for name, ls in ls_by_factor.items()}

fig, ax = plt.subplots(figsize=(9.5, 4.6))
names = list(stats.keys())
sharpes = [stats[n][1] for n in names]
colors = [FACTOR_COLORS[n] for n in names]
bars = ax.bar(names, sharpes, color=colors, edgecolor=PALETTE["spine"], linewidth=0.6)
for bar, v in zip(bars, sharpes):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (0.05 if bar.get_height() >= 0 else -0.15),
            f"{v:+.2f}", ha="center", va="bottom" if v >= 0 else "top",
            fontsize=10, color=INK, fontweight="bold")
ax.axhline(0, color=MUTED, linestyle="--", linewidth=1)
ax.set_title("Annualized Sharpe ratio of each long-short factor (crypto, 2020-2025)")
ax.set_ylabel("Sharpe ratio (gross of costs)")
y_lim = max(abs(v) for v in sharpes if not np.isnan(v)) * 1.4
ax.set_ylim(-y_lim, y_lim)
save_fig(fig, "crypto_factor_sharpe.png")

# ---------------------------------------------------------------------------
# Chart 7 — Drawdowns
# ---------------------------------------------------------------------------

print("Building chart 7 — drawdowns...")
fig, ax = plt.subplots(figsize=(9.5, 4.6))
for name, ls in ls_by_factor.items():
    cum = (1 + ls).cumprod()
    dd = (cum / cum.cummax() - 1) * 100
    ax.fill_between(dd.index, dd.values, 0, color=FACTOR_COLORS[name], alpha=0.18)
    ax.plot(dd.index, dd.values, color=FACTOR_COLORS[name], linewidth=1.8,
            label=f"{name}  max DD {dd.min():.0f}%")
ax.axhline(0, color=MUTED, linestyle="-", linewidth=0.6)
ax.set_title("Long-short drawdowns: crypto factors swing 30-50% routinely")
ax.set_xlabel("Date")
ax.set_ylabel("Drawdown from peak (%)")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
ax.legend(frameon=False, loc="lower left")
save_fig(fig, "crypto_factor_drawdowns.png")

# ---------------------------------------------------------------------------
# Chart 8 — Equity vs crypto IC comparison
# ---------------------------------------------------------------------------

print("Building chart 8 — equity vs crypto IC bars...")
# Equity ICs from the S&P 500 post for direct comparison
sp500_ic = {
    "Momentum (12-1)":     0.016,
    "Low volatility":      -0.020,
    "Short-term reversal": 0.019,
}
crypto_ic = {n: float(ic.mean()) for n, ic in ic_by_factor.items()}

fig, ax = plt.subplots(figsize=(9.5, 4.8))
x = np.arange(len(sp500_ic))
width = 0.38
b1 = ax.bar(x - width / 2, [sp500_ic[n] for n in sp500_ic], width,
            label="S&P 500 (2013-2018)", color=GOLD,
            edgecolor=PALETTE["spine"], linewidth=0.6)
b2 = ax.bar(x + width / 2, [crypto_ic[n] for n in sp500_ic], width,
            label="Crypto (2020 – today)", color=ACCENT,
            edgecolor=PALETTE["spine"], linewidth=0.6)
for bars, vals in [(b1, [sp500_ic[n] for n in sp500_ic]),
                   (b2, [crypto_ic[n] for n in sp500_ic])]:
    for bar, v in zip(bars, vals):
        offset = 0.003 if v >= 0 else -0.008
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + offset,
                f"{v:+.3f}", ha="center", va="bottom" if v >= 0 else "top",
                fontsize=9, color=INK)
ax.set_xticks(x)
ax.set_xticklabels(list(sp500_ic.keys()))
ax.axhline(0, color=MUTED, linestyle="--", linewidth=1)
ax.set_ylabel("Mean cross-sectional IC")
ax.set_title("Equity vs crypto: do the same factors work in both?")
ax.legend(frameon=False, loc="lower right")
save_fig(fig, "crypto_vs_equity_ic.png")

# ---------------------------------------------------------------------------
# Metrics export
# ---------------------------------------------------------------------------

metrics = {
    "data": {
        "n_rows_filtered": int(len(df)),
        "n_tickers": int(n_tickers),
        "date_start": str(date_start),
        "date_end": str(date_end),
        "n_months": int(len(ret_m)),
        "winsorize_bounds": [round(WINS[0], 4), round(WINS[1], 4)],
    },
    "universe_total_return_pct": round((ew_cum.iloc[-1] - 1) * 100, 1),
    "btc_total_return_pct": round(((1 + ret_d.get("BTC-USD")).cumprod().iloc[-1] - 1) * 100, 1)
        if "BTC-USD" in close.columns else None,
    "factors": {},
    "vs_sp500": {
        "Momentum (12-1)":     {"sp500_ic": 0.016, "crypto_ic": round(crypto_ic["Momentum (12-1)"], 4)},
        "Low volatility":      {"sp500_ic": -0.020, "crypto_ic": round(crypto_ic["Low volatility"], 4)},
        "Short-term reversal": {"sp500_ic": 0.019, "crypto_ic": round(crypto_ic["Short-term reversal"], 4)},
    },
}
for name in factors:
    ic = ic_by_factor[name]
    ls = ls_by_factor[name]
    ann_ret, sharpe, mdd = stats[name]
    t_stat = float(ic.mean() / (ic.std() / np.sqrt(len(ic)))) if len(ic) > 1 else float("nan")
    metrics["factors"][name] = {
        "mean_ic": round(float(ic.mean()), 4),
        "ic_t_stat": round(t_stat, 2),
        "ic_n_months": int(len(ic)),
        "annualized_return_pct": round(ann_ret * 100, 2),
        "annualized_sharpe": round(sharpe, 2),
        "max_drawdown_pct": round(mdd * 100, 2),
        "best_month_pct": round(float(ls.max()) * 100, 2),
        "worst_month_pct": round(float(ls.min()) * 100, 2),
    }

(OUTPUT / "crypto_factor_metrics.json").write_text(json.dumps(metrics, indent=2))
print(f"\nMetrics written to {(OUTPUT / 'crypto_factor_metrics.json').relative_to(ROOT)}")
print("Done.")
