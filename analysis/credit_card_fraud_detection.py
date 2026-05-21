"""
Detecting Credit Card Fraud — The Metric That Tells the Truth

Analyzes the mlg-ulb/creditcardfraud Kaggle dataset (284,807 European
cardholder transactions, 492 frauds = 0.173% fraud rate).

The methodological angle: at this class imbalance, ROC AUC and accuracy
are both misleading. PR-AUC (precision-recall AUC) and precision-at-N
are the metrics that actually tell you whether your fraud model works
in practice. We train LR and RF and evaluate both views.

Outputs:
  - assets/*.png         Charts referenced inline in the blog HTML
  - output/credit_card_fraud_metrics.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Paths and styling
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "datasets" / "creditcard.csv"

parser = argparse.ArgumentParser(description="Credit card fraud detection blog analysis")
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
        "accent":  "#34D399",
        "gold":    "#FBBF24",
        "danger":  "#F87171",
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
        "accent":  "#2563EB",
        "gold":    "#F59E0B",
        "danger":  "#DC2626",
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
DANGER = PALETTE["danger"]
GOLD = PALETTE["gold"]
PURPLE = PALETTE["purple"]
CYAN = PALETTE["cyan"]
BG = PALETTE["bg"]

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
# Load
# ---------------------------------------------------------------------------

print("Loading credit card fraud dataset (~150 MB)...")
df = pd.read_csv(DATA_PATH)
print(f"  rows={len(df):,}  cols={df.shape[1]}")

n_total = int(len(df))
n_fraud = int(df["Class"].sum())
fraud_rate = float(df["Class"].mean())
print(f"  fraud={n_fraud:,}  fraud_rate={fraud_rate:.4%}")
print(f"  trivial 'always predict legitimate' accuracy: {(1-fraud_rate):.4%}")

# ---------------------------------------------------------------------------
# Chart 1: Class imbalance with the trivial-classifier annotation
# ---------------------------------------------------------------------------

print("Building chart 1 — class imbalance...")
fig, ax = plt.subplots(figsize=(8.5, 4.3))
sizes = np.array([n_total - n_fraud, n_fraud])
labels = ["Legitimate", "Fraud"]
colors = [ACCENT, DANGER]
ax.barh([0], [sizes[0]], color=ACCENT, edgecolor=PALETTE["spine"], linewidth=0.6, label=f"Legitimate  {sizes[0]:,}  (99.827%)")
ax.barh([0], [sizes[1]], left=sizes[0], color=DANGER, edgecolor=PALETTE["spine"], linewidth=0.6, label=f"Fraud  {sizes[1]:,}  (0.173%)")
ax.set_yticks([])
ax.set_xlim(0, n_total)
ax.set_xlabel("Transaction count (48-hour window)")
ax.set_title("Class imbalance: 99.83% legitimate, 0.17% fraud")
ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1, -0.15), ncol=2)
ax.spines["bottom"].set_visible(True)
fig.subplots_adjust(bottom=0.32)
# Annotation arrow pointing at the tiny fraud slice
ax.annotate(
    "← 492 transactions\n(barely visible)",
    xy=(sizes[0] + sizes[1] / 2, 0.45),
    xytext=(sizes[0] - 60000, 0.55),
    fontsize=10, color=DANGER, ha="right",
    arrowprops=dict(arrowstyle="->", color=DANGER, lw=1),
)
save_fig(fig, "fraud_class_imbalance.png")

# ---------------------------------------------------------------------------
# Chart 2: Fraud rate by hour of day
# ---------------------------------------------------------------------------

print("Building chart 2 — fraud rate by hour-of-day proxy...")
# Time is seconds from start of the 48-hour window. Day modulo 86400 gives
# an hour-of-day approximation under the assumption that data starts at the
# same time on each of the two days.
df["_hour"] = ((df["Time"] % 86400) // 3600).astype(int)
by_hour = df.groupby("_hour")["Class"].agg(["mean", "count"])
fig, ax = plt.subplots(figsize=(9, 4.6))
bars = ax.bar(by_hour.index, by_hour["mean"] * 100, color=PURPLE,
              edgecolor=PALETTE["spine"], linewidth=0.6)
ax.axhline(fraud_rate * 100, color=MUTED, linestyle="--", linewidth=1.2,
           label=f"Overall fraud rate ({fraud_rate*100:.3f}%)")
ax.set_title("Fraud rate spikes overnight — when human review thins")
ax.set_xlabel("Hour of day (transaction-time modulo 24h)")
ax.set_ylabel("Fraud rate (%)")
ax.set_xticks(range(0, 24, 2))
ax.set_ylim(0, by_hour["mean"].max() * 100 * 1.15)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}%"))
ax.legend(frameon=False, loc="upper right")
save_fig(fig, "fraud_by_hour.png")

# ---------------------------------------------------------------------------
# Chart 3: Amount distributions (log-x) — the counterintuitive finding
# ---------------------------------------------------------------------------

print("Building chart 3 — amount distributions (log scale)...")
fig, ax = plt.subplots(figsize=(9, 4.6))
amt_legit = df.loc[df["Class"] == 0, "Amount"].clip(lower=0.01)
amt_fraud = df.loc[df["Class"] == 1, "Amount"].clip(lower=0.01)
bins = np.logspace(np.log10(0.01), np.log10(25700), 60)
ax.hist(amt_legit, bins=bins, color=ACCENT, alpha=0.55,
        edgecolor=PALETTE["spine"], linewidth=0.3, density=True,
        label=f"Legitimate  median ${amt_legit.median():.2f}")
ax.hist(amt_fraud, bins=bins, color=DANGER, alpha=0.65,
        edgecolor=PALETTE["spine"], linewidth=0.3, density=True,
        label=f"Fraud  median ${amt_fraud.median():.2f}")
ax.set_xscale("log")
ax.set_title("Fraud isn't where you'd guess: median fraud is $9, median legit is $22")
ax.set_xlabel("Transaction amount (USD, log scale)")
ax.set_ylabel("Density")
ax.legend(frameon=False, loc="upper right")
save_fig(fig, "fraud_amount_distribution.png")

# ---------------------------------------------------------------------------
# Modeling
# ---------------------------------------------------------------------------

print("Preparing feature matrix...")
feature_cols = [c for c in df.columns if c not in {"Class", "_hour"}]
X = df[feature_cols].values.astype(float)
y = df["Class"].astype(int).values
print(f"  X shape: {X.shape}  features: {feature_cols[:6]}... + Amount")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
print(f"  train={len(X_train):,}  test={len(X_test):,}  test_fraud={int(y_test.sum()):,}")

print("Training logistic regression (class_weight=balanced)...")
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
lr = LogisticRegression(max_iter=4000, C=1.0, class_weight="balanced")
lr.fit(X_train_s, y_train)
lr_probs = lr.predict_proba(X_test_s)[:, 1]

print("Training random forest (class_weight=balanced, ~60s)...")
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_leaf=2,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42,
)
rf.fit(X_train, y_train)
rf_probs = rf.predict_proba(X_test)[:, 1]

# ---------------------------------------------------------------------------
# Headline metrics: ROC-AUC, PR-AUC, precision@top-N, recall at low-FPR
# ---------------------------------------------------------------------------

def threshold_for_top_k(probs: np.ndarray, k_frac: float) -> float:
    """The cutoff that flags the top k_frac of the test set as suspicious."""
    return float(np.quantile(probs, 1 - k_frac))


def evaluate(name: str, probs: np.ndarray, y_true: np.ndarray) -> dict:
    roc_auc = float(roc_auc_score(y_true, probs))
    pr_auc = float(average_precision_score(y_true, probs))
    # Precision/recall at top-0.5% (operationally — fraud queue capacity)
    top_thresh = threshold_for_top_k(probs, 0.005)
    top_preds = (probs >= top_thresh).astype(int)
    p_at_top = float(precision_score(y_true, top_preds, zero_division=0))
    r_at_top = float(recall_score(y_true, top_preds))
    # At threshold 0.5 (naive default)
    preds_5 = (probs >= 0.5).astype(int)
    cm_5 = confusion_matrix(y_true, preds_5)
    p_5 = float(precision_score(y_true, preds_5, zero_division=0))
    r_5 = float(recall_score(y_true, preds_5))
    f1_5 = float(f1_score(y_true, preds_5, zero_division=0))
    acc_5 = float((preds_5 == y_true).mean())
    print(f"  {name:24} roc_auc={roc_auc:.4f}  pr_auc={pr_auc:.4f}  "
          f"P@top0.5%={p_at_top:.3f}  R@top0.5%={r_at_top:.3f}  "
          f"acc@.5={acc_5:.4f}  P@.5={p_5:.3f}  R@.5={r_5:.3f}")
    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "precision_at_top_0_5pct": p_at_top,
        "recall_at_top_0_5pct": r_at_top,
        "accuracy_at_0_5": acc_5,
        "precision_at_0_5": p_5,
        "recall_at_0_5": r_5,
        "f1_at_0_5": f1_5,
        "confusion_at_0_5": cm_5,
        "threshold_top_0_5pct": top_thresh,
    }


print("Evaluating models:")
lr_eval = evaluate("Logistic regression", lr_probs, y_test)
rf_eval = evaluate("Random forest", rf_probs, y_test)

# ---------------------------------------------------------------------------
# Chart 4: ROC vs PR curves side-by-side (the methodology chart)
# ---------------------------------------------------------------------------

print("Building chart 4 — ROC vs PR side-by-side...")
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probs)
axes[0].plot(lr_fpr, lr_tpr, color=ACCENT, linewidth=2.5,
             label=f"Logistic regression (AUC = {lr_eval['roc_auc']:.3f})")
axes[0].plot(rf_fpr, rf_tpr, color=PURPLE, linewidth=2.5,
             label=f"Random forest (AUC = {rf_eval['roc_auc']:.3f})")
axes[0].plot([0, 1], [0, 1], color=MUTED, linestyle="--", linewidth=1.2,
             label="Random (AUC = 0.5)")
axes[0].set_title("ROC curve: both models look ~excellent")
axes[0].set_xlabel("False positive rate")
axes[0].set_ylabel("True positive rate")
axes[0].set_xlim(0, 1)
axes[0].set_ylim(0, 1.02)
axes[0].legend(frameon=False, loc="lower right")

lr_p, lr_r, _ = precision_recall_curve(y_test, lr_probs)
rf_p, rf_r, _ = precision_recall_curve(y_test, rf_probs)
axes[1].plot(lr_r, lr_p, color=ACCENT, linewidth=2.5,
             label=f"Logistic regression (PR-AUC = {lr_eval['pr_auc']:.3f})")
axes[1].plot(rf_r, rf_p, color=PURPLE, linewidth=2.5,
             label=f"Random forest (PR-AUC = {rf_eval['pr_auc']:.3f})")
axes[1].axhline(fraud_rate, color=MUTED, linestyle="--", linewidth=1.2,
                label=f"Random (PR-AUC ≈ {fraud_rate:.3f})")
axes[1].set_title("PR curve: random forest dominates LR cleanly")
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].set_xlim(0, 1)
axes[1].set_ylim(0, 1.02)
axes[1].legend(frameon=False, loc="upper right")

fig.suptitle("Same two models, two metrics — only one shows real model difference",
             color=NAVY, fontweight="bold", fontsize=14, y=1.02)
fig.tight_layout()
save_fig(fig, "fraud_roc_vs_pr.png")

# ---------------------------------------------------------------------------
# Chart 5: Model comparison metric bars
# ---------------------------------------------------------------------------

print("Building chart 5 — model comparison metrics...")
metric_names = ["ROC AUC", "PR-AUC", "Precision\n@ top 0.5%", "Recall\n@ top 0.5%"]
lr_vals = [lr_eval["roc_auc"], lr_eval["pr_auc"],
           lr_eval["precision_at_top_0_5pct"], lr_eval["recall_at_top_0_5pct"]]
rf_vals = [rf_eval["roc_auc"], rf_eval["pr_auc"],
           rf_eval["precision_at_top_0_5pct"], rf_eval["recall_at_top_0_5pct"]]
x = np.arange(len(metric_names))
width = 0.38
fig, ax = plt.subplots(figsize=(9.5, 4.8))
b1 = ax.bar(x - width / 2, lr_vals, width, label="Logistic regression",
            color=ACCENT, edgecolor=PALETTE["spine"], linewidth=0.6)
b2 = ax.bar(x + width / 2, rf_vals, width, label="Random forest",
            color=PURPLE, edgecolor=PALETTE["spine"], linewidth=0.6)
for bars, vals in [(b1, lr_vals), (b2, rf_vals)]:
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9, color=INK)
ax.set_xticks(x)
ax.set_xticklabels(metric_names)
ax.set_ylim(0, 1.12)
ax.set_ylabel("Score")
ax.set_title("Random forest's edge is invisible on ROC AUC, obvious on PR-AUC")
ax.legend(frameon=False, loc="upper right")
save_fig(fig, "fraud_model_comparison.png")

# ---------------------------------------------------------------------------
# Chart 6: Confusion matrices at TWO thresholds for the better model (RF)
# ---------------------------------------------------------------------------

print("Building chart 6 — confusion matrices at two thresholds...")
# Threshold 1: default 0.5 (the naive choice)
preds_default = (rf_probs >= 0.5).astype(int)
cm_default = confusion_matrix(y_test, preds_default)
# Threshold 2: tuned to flag top 0.5% (operational choice)
preds_top = (rf_probs >= rf_eval["threshold_top_0_5pct"]).astype(int)
cm_top = confusion_matrix(y_test, preds_top)

cm_cmap = "mako" if args.theme == "dark" else "Blues"
cm_annot_color = "#F8FAFC" if args.theme == "dark" else NAVY

fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
for ax_i, (cm, title) in enumerate([
    (cm_default, f"Random forest @ threshold 0.5\nP={rf_eval['precision_at_0_5']:.2f}  R={rf_eval['recall_at_0_5']:.2f}"),
    (cm_top, f"Random forest @ top-0.5% threshold\nP={rf_eval['precision_at_top_0_5pct']:.2f}  R={rf_eval['recall_at_top_0_5pct']:.2f}"),
]):
    tn, fp, fn, tp = cm.ravel()
    matrix = np.array([[tn, fp], [fn, tp]])
    labels = np.array([
        [f"True negatives\n{tn:,}", f"False positives\n{fp:,}"],
        [f"False negatives\n{fn:,}", f"True positives\n{tp:,}"],
    ])
    sns.heatmap(
        matrix,
        annot=labels,
        fmt="",
        cmap=cm_cmap,
        cbar=False,
        linewidths=1,
        linecolor=BG,
        xticklabels=["Predicted legit", "Predicted fraud"],
        yticklabels=["Actual legit", "Actual fraud"],
        ax=axes[ax_i],
        annot_kws={"fontsize": 11, "color": cm_annot_color, "weight": "bold"},
    )
    axes[ax_i].set_title(title)

fig.suptitle("Same model, two thresholds: threshold tuning is the lever, not algorithm choice",
             color=NAVY, fontweight="bold", fontsize=13, y=1.02)
fig.tight_layout()
save_fig(fig, "fraud_confusion_matrices.png")

# ---------------------------------------------------------------------------
# Chart 7: Feature importance (random forest)
# ---------------------------------------------------------------------------

print("Building chart 7 — RF feature importance...")
fi = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=True)
top = fi.tail(12)
fig, ax = plt.subplots(figsize=(8.5, 6.2))
bars = ax.barh(top.index, top.values, color=PURPLE,
               edgecolor=PALETTE["spine"], linewidth=0.6)
for bar, v in zip(bars, top.values):
    ax.text(v + top.max() * 0.012, bar.get_y() + bar.get_height() / 2,
            f"{v:.3f}", va="center", fontsize=9, color=INK)
ax.set_title("Random forest feature importance — top 12")
ax.set_xlabel("Mean decrease in impurity")
ax.set_xlim(0, top.max() * 1.18)
save_fig(fig, "fraud_feature_importance.png")

# ---------------------------------------------------------------------------
# Chart 8: Cumulative gains curve (the fraud analyst's view)
# ---------------------------------------------------------------------------

print("Building chart 8 — cumulative gains curve...")

def gains_curve(probs, y_true):
    order = np.argsort(-probs)
    y_sorted = y_true[order]
    total_pos = y_sorted.sum()
    cum_pct_sample = np.arange(1, len(y_sorted) + 1) / len(y_sorted)
    cum_pct_caught = np.cumsum(y_sorted) / total_pos
    return cum_pct_sample, cum_pct_caught

lr_x, lr_y = gains_curve(lr_probs, y_test)
rf_x, rf_y = gains_curve(rf_probs, y_test)

fig, ax = plt.subplots(figsize=(8.5, 5))
ax.plot(lr_x * 100, lr_y * 100, color=ACCENT, linewidth=2.5,
        label="Logistic regression")
ax.plot(rf_x * 100, rf_y * 100, color=PURPLE, linewidth=2.5,
        label="Random forest")
ax.plot([0, 100], [0, 100], color=MUTED, linestyle="--", linewidth=1.2,
        label="Random review (no model)")
ax.fill_between(rf_x * 100, lr_y * 100, rf_y * 100,
                where=(rf_y >= lr_y), alpha=0.15, color=PURPLE,
                label="RF uplift over LR")
ax.set_title("Cumulative gains: reviewing the top 1% of RF-flagged catches ~75% of fraud")
ax.set_xlabel("% of transactions reviewed (ranked by model risk)")
ax.set_ylabel("% of fraud caught")
ax.set_xlim(0, 100)
ax.set_ylim(0, 102)
ax.set_xticks(range(0, 101, 10))
ax.set_yticks(range(0, 101, 20))
ax.legend(frameon=False, loc="lower right")
ax.axvline(1, color=DANGER, linestyle=":", linewidth=1.2, alpha=0.7)
ax.text(1.5, 8, "1%", color=DANGER, fontsize=10, alpha=0.8)
save_fig(fig, "fraud_cumulative_gains.png")

# Recall at top 1% (for the headline metric)
def recall_at_top(probs, y_true, frac):
    order = np.argsort(-probs)
    cutoff = int(len(probs) * frac)
    return float(y_true[order[:cutoff]].sum() / y_true.sum())

rf_recall_top1 = recall_at_top(rf_probs, y_test, 0.01)
lr_recall_top1 = recall_at_top(lr_probs, y_test, 0.01)
rf_recall_top5 = recall_at_top(rf_probs, y_test, 0.05)
lr_recall_top5 = recall_at_top(lr_probs, y_test, 0.05)
print(f"  recall @ top 1%:  LR={lr_recall_top1:.2%}  RF={rf_recall_top1:.2%}")
print(f"  recall @ top 5%:  LR={lr_recall_top5:.2%}  RF={rf_recall_top5:.2%}")

# ---------------------------------------------------------------------------
# Metrics export
# ---------------------------------------------------------------------------

metrics = {
    "n_total": n_total,
    "n_fraud": n_fraud,
    "fraud_rate_pct": round(fraud_rate * 100, 4),
    "trivial_classifier_accuracy_pct": round((1 - fraud_rate) * 100, 4),
    "amount_summary": {
        "median_legit": round(float(amt_legit.median()), 2),
        "median_fraud": round(float(amt_fraud.median()), 2),
        "mean_legit": round(float(amt_legit.mean()), 2),
        "mean_fraud": round(float(amt_fraud.mean()), 2),
    },
    "n_test": int(len(y_test)),
    "n_test_fraud": int(y_test.sum()),
    "models": {
        "logistic_regression": {
            "roc_auc": round(lr_eval["roc_auc"], 4),
            "pr_auc": round(lr_eval["pr_auc"], 4),
            "accuracy_at_0_5": round(lr_eval["accuracy_at_0_5"], 4),
            "precision_at_0_5": round(lr_eval["precision_at_0_5"], 4),
            "recall_at_0_5": round(lr_eval["recall_at_0_5"], 4),
            "f1_at_0_5": round(lr_eval["f1_at_0_5"], 4),
            "precision_at_top_0_5pct": round(lr_eval["precision_at_top_0_5pct"], 4),
            "recall_at_top_0_5pct": round(lr_eval["recall_at_top_0_5pct"], 4),
            "recall_at_top_1pct": round(lr_recall_top1, 4),
            "recall_at_top_5pct": round(lr_recall_top5, 4),
        },
        "random_forest": {
            "roc_auc": round(rf_eval["roc_auc"], 4),
            "pr_auc": round(rf_eval["pr_auc"], 4),
            "accuracy_at_0_5": round(rf_eval["accuracy_at_0_5"], 4),
            "precision_at_0_5": round(rf_eval["precision_at_0_5"], 4),
            "recall_at_0_5": round(rf_eval["recall_at_0_5"], 4),
            "f1_at_0_5": round(rf_eval["f1_at_0_5"], 4),
            "precision_at_top_0_5pct": round(rf_eval["precision_at_top_0_5pct"], 4),
            "recall_at_top_0_5pct": round(rf_eval["recall_at_top_0_5pct"], 4),
            "recall_at_top_1pct": round(rf_recall_top1, 4),
            "recall_at_top_5pct": round(rf_recall_top5, 4),
            "confusion_at_0_5": {
                "tn": int(rf_eval["confusion_at_0_5"][0, 0]),
                "fp": int(rf_eval["confusion_at_0_5"][0, 1]),
                "fn": int(rf_eval["confusion_at_0_5"][1, 0]),
                "tp": int(rf_eval["confusion_at_0_5"][1, 1]),
            },
            "threshold_top_0_5pct": round(rf_eval["threshold_top_0_5pct"], 4),
        },
    },
    "rf_top_features": [
        {"feature": k, "importance": round(float(v), 4)} for k, v in fi.tail(10)[::-1].items()
    ],
}
(OUTPUT / "credit_card_fraud_metrics.json").write_text(json.dumps(metrics, indent=2))
print(f"\nMetrics written to {(OUTPUT / 'credit_card_fraud_metrics.json').relative_to(ROOT)}")
print("Done.")
