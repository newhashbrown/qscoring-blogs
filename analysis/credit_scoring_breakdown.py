"""
Credit Scoring Models — Data-Driven Breakdown

Analyzes the laotse/credit-risk-dataset (32,581 loan applications) to
produce charts and metrics for the QScoring blog post.

Outputs:
  - assets/*.png         Charts referenced inline in the blog HTML
  - output/metrics.json  Numbers the HTML template substitutes in
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Paths and styling
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "datasets" / "credit_risk_dataset.csv"

parser = argparse.ArgumentParser(description="Credit scoring blog analysis")
parser.add_argument("--theme", choices=["light", "dark"], default="light")
parser.add_argument("--out-charts", type=str, default=None,
                    help="Override chart output dir (default: ./assets for light, ignored otherwise)")
parser.add_argument("--out-data", type=str, default=None,
                    help="Override metrics.json output dir (default: ./output)")
args = parser.parse_args()

if args.theme == "dark":
    PALETTE = {
        "bg":      "#0A0E17",   # qscoring --bg
        "bg_card": "#141c2e",   # qscoring --bg-card
        "title":   "#F8FAFC",   # --text
        "text":    "#CBD5E1",   # --text-dim
        "muted":   "#778999",   # --text-muted
        "accent":  "#34D399",   # --signal-buy (green)
        "gold":    "#FBBF24",   # --gold-bright
        "danger":  "#F87171",   # --signal-short
        "grid":    "rgba(203,213,225,0.10)",
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
        "accent":  "#2563EB",
        "gold":    "#F59E0B",
        "danger":  "#DC2626",
        "grid":    "#E2E8F0",
        "spine":   "#64748B",
    }
    default_charts_dir = ROOT / "assets"

ASSETS = Path(args.out_charts) if args.out_charts else default_charts_dir
OUTPUT = Path(args.out_data) if args.out_data else ROOT / "output"
ASSETS.mkdir(parents=True, exist_ok=True)
OUTPUT.mkdir(parents=True, exist_ok=True)

# Legacy aliases used throughout the chart code
NAVY = PALETTE["title"]
INK = PALETTE["text"]
MUTED = PALETTE["muted"]
ACCENT = PALETTE["accent"]
DANGER = PALETTE["danger"]
SAFE = PALETTE["accent"]  # in dark theme, "safe" = accent green
BG = PALETTE["bg"]

# In dark theme, recolor SAFE distinctly from ACCENT so the grade chart still
# uses three tiers. Reuse gold for the middle tier.
SAFE_TIER = PALETTE["accent"]
MID_TIER = PALETTE["gold"]
HIGH_TIER = PALETTE["danger"]

plt.rcParams.update(
    {
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
    }
)


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
# Load and clean
# ---------------------------------------------------------------------------

print("Loading credit-risk-dataset...")
df = pd.read_csv(DATA_PATH)
print(f"  rows={len(df):,}  cols={df.shape[1]}")

# Drop implausible ages (dataset has a few entries >100 — likely typos).
df = df[df["person_age"] <= 80].copy()
df = df[df["person_income"] <= df["person_income"].quantile(0.995)].copy()

# Median-fill the two columns with missing values.
df["person_emp_length"] = df["person_emp_length"].fillna(df["person_emp_length"].median())
df["loan_int_rate"] = df["loan_int_rate"].fillna(df["loan_int_rate"].median())

print(f"  cleaned rows={len(df):,}")

# ---------------------------------------------------------------------------
# Headline numbers
# ---------------------------------------------------------------------------

n_total = int(len(df))
n_default = int(df["loan_status"].sum())
default_rate = float(df["loan_status"].mean())
median_income = float(df["person_income"].median())
median_loan = float(df["loan_amnt"].median())
median_rate = float(df["loan_int_rate"].median())

print(
    f"  defaults={n_default:,}  default_rate={default_rate:.1%}  "
    f"median_income=${median_income:,.0f}"
)

# ---------------------------------------------------------------------------
# Chart 1: Default rate by loan grade
# ---------------------------------------------------------------------------

print("Building chart 1 — default rate by grade...")
grade_order = ["A", "B", "C", "D", "E", "F", "G"]
by_grade = (
    df.groupby("loan_grade")["loan_status"]
    .agg(["mean", "count"])
    .reindex(grade_order)
)

fig, ax = plt.subplots(figsize=(8.5, 4.5))
colors = [SAFE_TIER if r < 0.15 else MID_TIER if r < 0.35 else HIGH_TIER for r in by_grade["mean"]]
bars = ax.bar(by_grade.index, by_grade["mean"] * 100, color=colors, edgecolor=PALETTE["spine"], linewidth=0.6)
for bar, n in zip(bars, by_grade["count"]):
    h = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        h + 1.5,
        f"{h:.1f}%\nn={n:,}",
        ha="center",
        va="bottom",
        fontsize=9,
        color=INK,
    )
ax.set_title("Default rate climbs sharply with loan grade")
ax.set_xlabel("Loan grade (lender's internal risk tier)")
ax.set_ylabel("Default rate (%)")
ax.set_ylim(0, max(by_grade["mean"]) * 100 + 18)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
save_fig(fig, "default_by_grade.png")

# ---------------------------------------------------------------------------
# Chart 2: Default rate by home ownership
# ---------------------------------------------------------------------------

print("Building chart 2 — default rate by home ownership...")
home_map = {"RENT": "Rent", "OWN": "Own", "MORTGAGE": "Mortgage", "OTHER": "Other"}
df["_home"] = df["person_home_ownership"].map(home_map).fillna(df["person_home_ownership"])
by_home = (
    df.groupby("_home")["loan_status"]
    .agg(["mean", "count"])
    .sort_values("mean", ascending=False)
)

fig, ax = plt.subplots(figsize=(8.5, 4.5))
bars = ax.barh(
    by_home.index, by_home["mean"] * 100, color=ACCENT, edgecolor=PALETTE["spine"], linewidth=0.6
)
for bar, n in zip(bars, by_home["count"]):
    w = bar.get_width()
    ax.text(w + 0.6, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%  (n={n:,})", va="center", fontsize=9, color=INK)
ax.set_title("Renters default at roughly twice the rate of homeowners")
ax.set_xlabel("Default rate (%)")
ax.invert_yaxis()
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
ax.set_xlim(0, by_home["mean"].max() * 100 + 8)
save_fig(fig, "default_by_home.png")

# ---------------------------------------------------------------------------
# Chart 3: Default rate by loan intent
# ---------------------------------------------------------------------------

print("Building chart 3 — default rate by loan intent...")
intent_map = {
    "PERSONAL": "Personal",
    "EDUCATION": "Education",
    "MEDICAL": "Medical",
    "VENTURE": "Venture",
    "HOMEIMPROVEMENT": "Home improvement",
    "DEBTCONSOLIDATION": "Debt consolidation",
}
df["_intent"] = df["loan_intent"].map(intent_map).fillna(df["loan_intent"])
by_intent = (
    df.groupby("_intent")["loan_status"]
    .agg(["mean", "count"])
    .sort_values("mean", ascending=False)
)

fig, ax = plt.subplots(figsize=(8.5, 4.5))
bars = ax.barh(by_intent.index, by_intent["mean"] * 100, color=NAVY, edgecolor=PALETTE["spine"], linewidth=0.6)
for bar, n in zip(bars, by_intent["count"]):
    w = bar.get_width()
    ax.text(w + 0.4, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%  (n={n:,})", va="center", fontsize=9, color=INK)
ax.set_title("Why the borrower wants money matters")
ax.set_xlabel("Default rate (%)")
ax.invert_yaxis()
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
ax.set_xlim(0, by_intent["mean"].max() * 100 + 6)
save_fig(fig, "default_by_intent.png")

# ---------------------------------------------------------------------------
# Chart 4: Income distribution by default status
# ---------------------------------------------------------------------------

print("Building chart 4 — income distribution by outcome...")
fig, ax = plt.subplots(figsize=(8.5, 4.5))
income_repaid = df.loc[df["loan_status"] == 0, "person_income"]
income_default = df.loc[df["loan_status"] == 1, "person_income"]
sns.kdeplot(income_repaid, ax=ax, color=SAFE, fill=True, alpha=0.25, label=f"Repaid (n={len(income_repaid):,})", linewidth=2)
sns.kdeplot(income_default, ax=ax, color=DANGER, fill=True, alpha=0.25, label=f"Defaulted (n={len(income_default):,})", linewidth=2)
ax.set_xlim(0, df["person_income"].quantile(0.97))
ax.set_title("Defaulters cluster at lower incomes — but the overlap is real")
ax.set_xlabel("Annual income (USD)")
ax.set_ylabel("Density")
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v/1000:.0f}k"))
ax.legend(frameon=False)
save_fig(fig, "income_distribution.png")

# ---------------------------------------------------------------------------
# Chart 5: Loan-to-income ratio vs default
# ---------------------------------------------------------------------------

print("Building chart 5 — loan-to-income ratio bins...")
df["lti_bin"] = pd.cut(
    df["loan_percent_income"],
    bins=[0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0],
    labels=["<5%", "5-10%", "10-15%", "15-20%", "20-30%", "30-50%", "50%+"],
    include_lowest=True,
)
by_lti = df.groupby("lti_bin", observed=True)["loan_status"].agg(["mean", "count"])

fig, ax = plt.subplots(figsize=(8.5, 4.5))
bars = ax.bar(by_lti.index.astype(str), by_lti["mean"] * 100, color=ACCENT, edgecolor=PALETTE["spine"], linewidth=0.6)
for bar, n in zip(bars, by_lti["count"]):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h + 1.5, f"{h:.0f}%\nn={n:,}", ha="center", va="bottom", fontsize=9, color=INK)
ax.set_title("Loan-to-income ratio is one of the strongest single signals")
ax.set_xlabel("Loan as % of annual income")
ax.set_ylabel("Default rate (%)")
ax.set_ylim(0, by_lti["mean"].max() * 100 + 15)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
save_fig(fig, "default_by_lti.png")

# ---------------------------------------------------------------------------
# Logistic regression scoring model
# ---------------------------------------------------------------------------

print("Training logistic regression scoring model...")
features_num = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
]
features_cat = ["person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file"]

X_num = df[features_num].copy()
X_cat = pd.get_dummies(df[features_cat], drop_first=True)
X = pd.concat([X_num, X_cat], axis=1).astype(float)
y = df["loan_status"].astype(int).values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

model = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
model.fit(X_train_s, y_train)

probs = model.predict_proba(X_test_s)[:, 1]
preds = (probs >= 0.5).astype(int)
auc = float(roc_auc_score(y_test, probs))
accuracy = float((preds == y_test).mean())
cm = confusion_matrix(y_test, preds)
tn, fp, fn, tp = (int(v) for v in cm.ravel())
precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
print(f"  AUC={auc:.3f}  accuracy={accuracy:.3f}  precision={precision:.3f}  recall={recall:.3f}")

# ---------------------------------------------------------------------------
# Chart 6: ROC curve
# ---------------------------------------------------------------------------

print("Building chart 6 — ROC curve...")
fpr, tpr, _ = roc_curve(y_test, probs)
fig, ax = plt.subplots(figsize=(6.5, 5.5))
ax.plot(fpr, tpr, color=ACCENT, linewidth=2.5, label=f"Model (AUC = {auc:.3f})")
ax.plot([0, 1], [0, 1], color=MUTED, linestyle="--", linewidth=1.2, label="Random baseline")
ax.fill_between(fpr, tpr, alpha=0.08, color=ACCENT)
ax.set_title("Model separates good vs bad loans well above chance")
ax.set_xlabel("False positive rate (good loans flagged as risky)")
ax.set_ylabel("True positive rate (defaults correctly caught)")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.02)
ax.legend(frameon=False, loc="lower right")
save_fig(fig, "roc_curve.png")

# ---------------------------------------------------------------------------
# Chart 7: Feature importance (standardized coefficients)
# ---------------------------------------------------------------------------

print("Building chart 7 — feature importance...")
coef = pd.Series(model.coef_[0], index=X.columns).sort_values()
top = pd.concat([coef.head(6), coef.tail(8)])
fig, ax = plt.subplots(figsize=(8.5, 6))
colors = [SAFE if v < 0 else DANGER for v in top.values]
bars = ax.barh(top.index, top.values, color=colors, edgecolor=PALETTE["spine"], linewidth=0.6)
ax.axvline(0, color=NAVY, linewidth=0.8)
for bar, v in zip(bars, top.values):
    offset = 0.04 if v >= 0 else -0.04
    align = "left" if v >= 0 else "right"
    ax.text(v + offset, bar.get_y() + bar.get_height() / 2, f"{v:+.2f}", va="center", ha=align, fontsize=9, color=INK)
ax.set_title("What the model actually weights")
ax.set_xlabel("Standardized coefficient  (←  lowers risk     raises risk  →)")
margin = max(abs(top.min()), abs(top.max())) * 0.25
ax.set_xlim(top.min() - margin, top.max() + margin)
save_fig(fig, "feature_importance.png")

# ---------------------------------------------------------------------------
# Chart 8: Confusion matrix
# ---------------------------------------------------------------------------

print("Building chart 8 — confusion matrix...")
fig, ax = plt.subplots(figsize=(6, 5))
matrix = np.array([[tn, fp], [fn, tp]])
labels = np.array([[f"True negatives\n{tn:,}", f"False positives\n{fp:,}"],
                   [f"False negatives\n{fn:,}", f"True positives\n{tp:,}"]])
cm_cmap = "mako" if args.theme == "dark" else "Blues"
cm_annot_color = "#F8FAFC" if args.theme == "dark" else NAVY
sns.heatmap(
    matrix,
    annot=labels,
    fmt="",
    cmap=cm_cmap,
    cbar=False,
    linewidths=1,
    linecolor=BG,
    xticklabels=["Predicted repay", "Predicted default"],
    yticklabels=["Actual repay", "Actual default"],
    ax=ax,
    annot_kws={"fontsize": 11, "color": cm_annot_color, "weight": "bold"},
)
ax.set_title("Confusion matrix at default threshold (0.5)")
save_fig(fig, "confusion_matrix.png")

# ---------------------------------------------------------------------------
# Metrics export for the blog template
# ---------------------------------------------------------------------------

metrics = {
    "n_total": n_total,
    "n_default": n_default,
    "default_rate_pct": round(default_rate * 100, 1),
    "median_income": int(median_income),
    "median_loan": int(median_loan),
    "median_rate_pct": round(median_rate, 1),
    "grade_default_rates": {
        g: {"rate_pct": round(by_grade.loc[g, "mean"] * 100, 1), "n": int(by_grade.loc[g, "count"])}
        for g in grade_order
    },
    "home_default_rates": {
        h: {"rate_pct": round(by_home.loc[h, "mean"] * 100, 1), "n": int(by_home.loc[h, "count"])}
        for h in by_home.index
    },
    "intent_default_rates": {
        i: {"rate_pct": round(by_intent.loc[i, "mean"] * 100, 1), "n": int(by_intent.loc[i, "count"])}
        for i in by_intent.index
    },
    "lti_default_rates": {
        str(k): {"rate_pct": round(by_lti.loc[k, "mean"] * 100, 1), "n": int(by_lti.loc[k, "count"])}
        for k in by_lti.index
    },
    "model": {
        "auc": round(auc, 3),
        "accuracy": round(accuracy, 3),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "test_size": int(len(y_test)),
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "top_positive_features": [
            {"feature": k, "coef": round(float(v), 3)} for k, v in coef.tail(5)[::-1].items()
        ],
        "top_negative_features": [
            {"feature": k, "coef": round(float(v), 3)} for k, v in coef.head(5).items()
        ],
    },
}
(OUTPUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
print(f"\nMetrics written to {(OUTPUT / 'metrics.json').relative_to(ROOT)}")
print("Done.")
