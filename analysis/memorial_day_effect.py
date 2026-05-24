"""
How the Stock Market Behaves Before and After Memorial Day — Event Study

Pulls daily ^GSPC (S&P 500 index) from Yahoo Finance over 1990-01-01 →
today, identifies Memorial Day (last Monday of May) each year, and
computes returns in a 5-trading-day window before and after the holiday.

Anchor convention:
  day  0 = trading day immediately BEFORE Memorial Day (Friday close)
  day -1..-5 = the five trading days that close the week into Friday
  day +1..+5 = the five trading days starting Tuesday after the holiday
  Memorial Monday itself is a closed-market day; it is not an "event day."

For each year we compute:
  - pre_5d_ret  = cumulative return over days t=-5..0
  - post_5d_ret = cumulative return over days t=0..+5
  - cum_path    = cumulative path from day -5 through +5 (for averaging)

We then compare these distributions against a baseline of 10,000 random
non-overlapping 5-day windows drawn from the same date range, to test
whether the Memorial Day window is meaningfully different from "any
random week of S&P 500 returns."

Outputs:
  - assets/memorial-day/*.png             4 charts
  - output/memorial_day_metrics.json      Headline metrics referenced by
                                          the blog body prose.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import date

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import yfinance as yf

# ---------------------------------------------------------------------------
# Paths and styling (mirrors sp500_factor_test.py palette/conventions)
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser(description="Memorial Day event-study blog")
parser.add_argument("--theme", choices=["light", "dark"], default="light")
parser.add_argument("--out-charts", type=str, default=None)
parser.add_argument("--out-data", type=str, default=None)
parser.add_argument("--start", type=str, default="1990-01-01")
parser.add_argument("--end", type=str, default=None,
                    help="End date (default = today). Use to pin reproducibility.")
parser.add_argument("--n-baseline", type=int, default=10_000,
                    help="Number of random 5-day baseline windows to draw.")
parser.add_argument("--seed", type=int, default=20260523)
args = parser.parse_args()

if args.theme == "dark":
    PALETTE = {
        "bg": "#0A0E17", "title": "#F8FAFC", "text": "#CBD5E1",
        "muted": "#778999", "accent": "#FBBF24", "gold": "#FBBF24",
        "green": "#34D399", "red": "#F87171", "purple": "#A78BFA",
        "cyan": "#22D3EE", "spine": "#2B3548",
    }
    default_charts_dir = ROOT / "assets-dark" / "memorial-day"
else:
    PALETTE = {
        "bg": "#FFFFFF", "title": "#0F172A", "text": "#1E293B",
        "muted": "#64748B", "accent": "#D97706", "gold": "#F59E0B",
        "green": "#059669", "red": "#DC2626", "purple": "#7C3AED",
        "cyan": "#0891B2", "spine": "#64748B",
    }
    default_charts_dir = ROOT / "assets" / "memorial-day"

ASSETS = Path(args.out_charts) if args.out_charts else default_charts_dir
OUTPUT = Path(args.out_data) if args.out_data else ROOT / "output"
ASSETS.mkdir(parents=True, exist_ok=True)
OUTPUT.mkdir(parents=True, exist_ok=True)

NAVY = PALETTE["title"]; INK = PALETTE["text"]; MUTED = PALETTE["muted"]
ACCENT = PALETTE["accent"]; GOLD = PALETTE["gold"]; GREEN = PALETTE["green"]
RED = PALETTE["red"]; PURPLE = PALETTE["purple"]; CYAN = PALETTE["cyan"]
BG = PALETTE["bg"]

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG,
    "axes.edgecolor": PALETTE["spine"], "axes.labelcolor": INK,
    "axes.titlecolor": NAVY, "axes.titleweight": "bold",
    "axes.titlesize": 14, "axes.labelsize": 11,
    "xtick.color": INK, "ytick.color": INK,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "DejaVu Sans", "figure.dpi": 140, "savefig.dpi": 140,
    "savefig.bbox": "tight", "savefig.facecolor": BG,
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
# Load S&P 500 daily closes
# ---------------------------------------------------------------------------

end = args.end or date.today().isoformat()
print(f"Loading ^GSPC daily closes {args.start} -> {end}...")
raw = yf.download("^GSPC", start=args.start, end=end,
                  progress=False, auto_adjust=True)
if raw.empty:
    raise SystemExit("yfinance returned no data; check network connectivity.")

# yfinance returns a MultiIndex column layout when downloading via `download()`.
# Normalize down to a flat Series of daily closes.
close = raw["Close"]
if isinstance(close, pd.DataFrame):
    close = close.iloc[:, 0]
close = close.dropna().sort_index()
ret = close.pct_change().dropna()
print(f"  trading days: {len(close):,}  range={close.index.min().date()} to {close.index.max().date()}")


# ---------------------------------------------------------------------------
# Memorial Day calendar
# ---------------------------------------------------------------------------

def memorial_day(year: int) -> date:
    """Last Monday of May for the given year."""
    d = date(year, 5, 31)
    # weekday(): Monday = 0
    return date(year, 5, 31 - d.weekday())


years = sorted({d.year for d in close.index.date if d.month >= 5 and d.year >= int(args.start[:4])})
md_dates = [memorial_day(y) for y in years]
# Drop the most recent year if Memorial Day hasn't happened yet OR the
# post window doesn't fit fully into the data we have.
trading_days = pd.DatetimeIndex(close.index.date)


def trading_idx_before(d: date) -> int | None:
    """Index into `close` of the last trading day strictly before `d`."""
    arr = close.index.date
    mask = arr < d
    if not mask.any():
        return None
    return int(np.where(mask)[0][-1])


def event_window(d: date, pre: int = 5, post: int = 5) -> tuple[int, np.ndarray] | None:
    """
    Returns (anchor_idx, returns_array) where:
      - anchor_idx = index of the trading day immediately BEFORE Memorial Day
      - returns_array has length pre+post; the first `pre` entries are the
        returns ending at anchor (incl. anchor's own day return), and the
        last `post` are returns starting the day after anchor.

    Returns None if the window doesn't fit fully into the loaded series.
    """
    anchor = trading_idx_before(d)
    if anchor is None:
        return None
    if anchor - pre < 0 or anchor + post >= len(close):
        return None
    # ret aligns to dates in `close` minus the first row.  Easier to slice
    # by date positions in `close` directly.
    pre_slice = close.iloc[anchor - pre + 1 : anchor + 1].values \
                / close.iloc[anchor - pre : anchor].values - 1
    post_slice = close.iloc[anchor + 1 : anchor + 1 + post].values \
                 / close.iloc[anchor : anchor + post].values - 1
    return anchor, np.concatenate([pre_slice, post_slice])


pre_n, post_n = 5, 5
records: list[dict] = []
for md in md_dates:
    win = event_window(md, pre_n, post_n)
    if win is None:
        print(f"  skipping {md} — window does not fit")
        continue
    anchor, daily_rets = win
    pre_cum = float(np.prod(1 + daily_rets[:pre_n]) - 1)
    post_cum = float(np.prod(1 + daily_rets[pre_n:]) - 1)
    records.append({
        "year": md.year,
        "memorial_day": md.isoformat(),
        "anchor_trading_day": close.index[anchor].date().isoformat(),
        "pre_5d_ret": pre_cum,
        "post_5d_ret": post_cum,
        "daily_rets": daily_rets.tolist(),
    })

events = pd.DataFrame(records)
print(f"  Memorial Day observations: {len(events)}")
print(f"  years: {events['year'].min()} ... {events['year'].max()}")


# ---------------------------------------------------------------------------
# Headline statistics + baseline
# ---------------------------------------------------------------------------

def summary(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    n = len(x)
    mean = float(np.mean(x))
    median = float(np.median(x))
    std = float(np.std(x, ddof=1))
    se = std / np.sqrt(n)
    t = mean / se if se > 0 else float("nan")
    p = float(2 * (1 - stats.t.cdf(abs(t), df=n - 1))) if n > 1 else float("nan")
    hit = float(np.mean(x > 0))
    return {"n": n, "mean": mean, "median": median, "std": std,
            "se": se, "t_stat": float(t), "p_value": p, "hit_rate": hit}


pre_stats = summary(events["pre_5d_ret"].values)
post_stats = summary(events["post_5d_ret"].values)
print(f"  pre  mean={pre_stats['mean']*100:+.3f}%  t={pre_stats['t_stat']:+.2f}  hit={pre_stats['hit_rate']*100:.0f}%")
print(f"  post mean={post_stats['mean']*100:+.3f}%  t={post_stats['t_stat']:+.2f}  hit={post_stats['hit_rate']*100:.0f}%")

# Baseline: random non-overlapping 5-day windows across the full date range.
# We sample `n_baseline` anchor positions uniformly from valid indices and
# compute the cumulative 5-day return starting at each.
rng = np.random.default_rng(args.seed)
valid_lo = pre_n
valid_hi = len(close) - post_n - 1
anchor_choices = rng.integers(low=valid_lo, high=valid_hi, size=args.n_baseline)

baseline_pre = np.empty(args.n_baseline)
baseline_post = np.empty(args.n_baseline)
for i, a in enumerate(anchor_choices):
    pre_w = close.iloc[a - pre_n + 1 : a + 1].values \
            / close.iloc[a - pre_n : a].values - 1
    post_w = close.iloc[a + 1 : a + 1 + post_n].values \
             / close.iloc[a : a + post_n].values - 1
    baseline_pre[i] = np.prod(1 + pre_w) - 1
    baseline_post[i] = np.prod(1 + post_w) - 1

baseline_pre_stats = summary(baseline_pre)
baseline_post_stats = summary(baseline_post)
print(f"  baseline pre  mean={baseline_pre_stats['mean']*100:+.3f}%  hit={baseline_pre_stats['hit_rate']*100:.1f}%")
print(f"  baseline post mean={baseline_post_stats['mean']*100:+.3f}%  hit={baseline_post_stats['hit_rate']*100:.1f}%")

# Two-sample t-test: Memorial-Day window vs random baseline.
t_pre, p_pre = stats.ttest_ind(events["pre_5d_ret"].values, baseline_pre,
                                equal_var=False)
t_post, p_post = stats.ttest_ind(events["post_5d_ret"].values, baseline_post,
                                  equal_var=False)


# ---------------------------------------------------------------------------
# Chart 1 — Average cumulative path across [-5, +5] with +/-1 SE band
# ---------------------------------------------------------------------------

print("Building chart 1 — average event-study path...")
# Convert each event's per-day returns into a cumulative path indexed at
# day 0 = end of trading day BEFORE Memorial Day. Path includes day -5
# (start), accumulates day -5..0 (5 returns) ending at day 0 = 1.0, then
# extends day +1..+5 (5 more returns).
# We re-anchor each path so day 0 = 1.0 to study deviations around the event.
paths = []
for _, row in events.iterrows():
    daily = np.asarray(row["daily_rets"], dtype=float)
    # cumulative growth from day -5 (start) using the 5 pre-returns and 5 post-returns
    growth = np.concatenate([[1.0], np.cumprod(1 + daily)])
    # Re-index so day 0 = 1.0 (anchor)
    growth = growth / growth[pre_n]
    paths.append(growth)

paths = np.vstack(paths)  # shape: (n_events, pre_n + post_n + 1)
mean_path = paths.mean(axis=0)
se_path = paths.std(axis=0, ddof=1) / np.sqrt(paths.shape[0])
days_axis = np.arange(-pre_n, post_n + 1)

fig, ax = plt.subplots(figsize=(9.5, 4.8))
ax.fill_between(days_axis, (mean_path - se_path - 1) * 100,
                (mean_path + se_path - 1) * 100,
                color=ACCENT, alpha=0.18, label="±1 standard error")
ax.plot(days_axis, (mean_path - 1) * 100, color=ACCENT, linewidth=2.8,
        marker="o", markersize=5, label=f"Average path ({paths.shape[0]} years)")
ax.axhline(0, color=MUTED, linestyle="--", linewidth=0.8)
ax.axvline(0, color=MUTED, linestyle=":", linewidth=1)
ax.text(0.05, 0.95,
        "Day 0 = trading day BEFORE Memorial Day (Friday close)\n"
        "Memorial Monday is a market holiday — no return for that day",
        transform=ax.transAxes, va="top", color=MUTED, fontsize=9)
ax.set_title("Average S&P 500 path around Memorial Day, 1990–2025")
ax.set_xlabel("Trading days from Friday-before-Memorial-Day")
ax.set_ylabel("Cumulative return vs day 0")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.2f}%"))
ax.legend(frameon=False, loc="lower right")
save_fig(fig, "memorial_day_event_study.png")


# ---------------------------------------------------------------------------
# Chart 2 — Per-year pre vs post 5-day return
# ---------------------------------------------------------------------------

print("Building chart 2 — yearly pre vs post returns...")
fig, ax = plt.subplots(figsize=(11, 4.8))
yrs = events["year"].values
x = np.arange(len(yrs))
w = 0.4
ax.bar(x - w/2, events["pre_5d_ret"].values * 100, width=w,
       color=CYAN, alpha=0.85, label="Pre (5 days before)")
ax.bar(x + w/2, events["post_5d_ret"].values * 100, width=w,
       color=ACCENT, alpha=0.85, label="Post (5 days after)")
ax.axhline(0, color=MUTED, linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels([str(y) for y in yrs], rotation=70, fontsize=8)
ax.set_title("S&P 500 returns in the 5 trading days before and after Memorial Day, by year")
ax.set_xlabel("Year")
ax.set_ylabel("Cumulative 5-day return")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.0f}%"))
ax.legend(frameon=False, loc="upper left")
save_fig(fig, "memorial_day_yearly_pre_post.png")


# ---------------------------------------------------------------------------
# Chart 3 — Distribution vs random baseline
# ---------------------------------------------------------------------------

print("Building chart 3 — distribution vs baseline...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)

bins = np.linspace(-0.10, 0.10, 41)

ax1.hist(np.clip(baseline_pre, -0.10, 0.10), bins=bins, color=MUTED,
         alpha=0.45, density=True, label="Random 5-day windows")
ax1.hist(np.clip(events["pre_5d_ret"].values, -0.10, 0.10), bins=bins,
         color=CYAN, alpha=0.85, density=True,
         label=f"Pre-Memorial Day  (n={pre_stats['n']})")
ax1.axvline(baseline_pre_stats["mean"], color=MUTED, linestyle="--", linewidth=1.2)
ax1.axvline(pre_stats["mean"], color=CYAN, linestyle="-", linewidth=2)
ax1.set_title("Pre-window: 5 days before")
ax1.set_xlabel("Cumulative 5-day return")
ax1.set_ylabel("Density")
ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:+.0f}%"))
ax1.legend(frameon=False, loc="upper left", fontsize=9)

ax2.hist(np.clip(baseline_post, -0.10, 0.10), bins=bins, color=MUTED,
         alpha=0.45, density=True, label="Random 5-day windows")
ax2.hist(np.clip(events["post_5d_ret"].values, -0.10, 0.10), bins=bins,
         color=ACCENT, alpha=0.85, density=True,
         label=f"Post-Memorial Day  (n={post_stats['n']})")
ax2.axvline(baseline_post_stats["mean"], color=MUTED, linestyle="--", linewidth=1.2)
ax2.axvline(post_stats["mean"], color=ACCENT, linestyle="-", linewidth=2)
ax2.set_title("Post-window: 5 days after")
ax2.set_xlabel("Cumulative 5-day return")
ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:+.0f}%"))
ax2.legend(frameon=False, loc="upper left", fontsize=9)

fig.suptitle("Memorial Day windows vs random 5-day windows",
             color=NAVY, fontweight="bold", fontsize=14, y=1.02)
save_fig(fig, "memorial_day_vs_baseline.png")


# ---------------------------------------------------------------------------
# Chart 4 — Hit-rate comparison
# ---------------------------------------------------------------------------

print("Building chart 4 — hit rates...")
fig, ax = plt.subplots(figsize=(8.5, 4.2))
labels = ["Pre window\n(5 days before)", "Post window\n(5 days after)"]
md_rates = [pre_stats["hit_rate"] * 100, post_stats["hit_rate"] * 100]
bl_rates = [baseline_pre_stats["hit_rate"] * 100, baseline_post_stats["hit_rate"] * 100]
x = np.arange(2)
w = 0.36
ax.bar(x - w/2, md_rates, width=w, color=ACCENT, alpha=0.9,
       label=f"Memorial Day window (n={pre_stats['n']})")
ax.bar(x + w/2, bl_rates, width=w, color=MUTED, alpha=0.7,
       label=f"Random 5-day windows (n={args.n_baseline:,})")
ax.axhline(50, color=MUTED, linestyle="--", linewidth=0.8)
for i, v in enumerate(md_rates):
    ax.text(i - w/2, v + 1.0, f"{v:.0f}%", ha="center", color=NAVY, fontsize=10, fontweight="bold")
for i, v in enumerate(bl_rates):
    ax.text(i + w/2, v + 1.0, f"{v:.0f}%", ha="center", color=MUTED, fontsize=10)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0, max(max(md_rates), max(bl_rates)) + 10)
ax.set_title("Hit rate — % of windows with positive cumulative return")
ax.set_ylabel("% positive")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
ax.legend(frameon=False, loc="upper right")
save_fig(fig, "memorial_day_hit_rate.png")


# ---------------------------------------------------------------------------
# Persist metrics
# ---------------------------------------------------------------------------

metrics = {
    "source": "Yahoo Finance ^GSPC (S&P 500 index) via yfinance",
    "data_range": {
        "start": close.index.min().date().isoformat(),
        "end": close.index.max().date().isoformat(),
        "trading_days": int(len(close)),
    },
    "window": {"pre_days": pre_n, "post_days": post_n,
               "anchor": "trading day immediately before Memorial Day"},
    "n_events": int(len(events)),
    "year_range": [int(events["year"].min()), int(events["year"].max())],
    "pre": pre_stats,
    "post": post_stats,
    "baseline": {
        "n_samples": int(args.n_baseline),
        "seed": int(args.seed),
        "pre": baseline_pre_stats,
        "post": baseline_post_stats,
    },
    "tests_vs_baseline": {
        "pre":  {"t_stat": float(t_pre),  "p_value": float(p_pre)},
        "post": {"t_stat": float(t_post), "p_value": float(p_post)},
    },
    "per_year": [
        {
            "year": int(r["year"]),
            "memorial_day": r["memorial_day"],
            "pre_5d_ret":  float(r["pre_5d_ret"]),
            "post_5d_ret": float(r["post_5d_ret"]),
        }
        for _, r in events.iterrows()
    ],
}

out_path = OUTPUT / "memorial_day_metrics.json"
out_path.write_text(json.dumps(metrics, indent=2))
print(f"  wrote {out_path.relative_to(ROOT)}")

print("\nDone.")
print(f"  pre  mean={pre_stats['mean']*100:+.3f}% (baseline {baseline_pre_stats['mean']*100:+.3f}%) "
      f"t_vs_baseline={t_pre:+.2f} p={p_pre:.3f}")
print(f"  post mean={post_stats['mean']*100:+.3f}% (baseline {baseline_post_stats['mean']*100:+.3f}%) "
      f"t_vs_baseline={t_post:+.2f} p={p_post:.3f}")
