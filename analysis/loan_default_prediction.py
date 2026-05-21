"""
Predicting Loan Defaults — Logistic Regression vs Random Forest

Analyzes the hemanthsai7/loandefault Kaggle dataset (67,463 loans,
9.25% default rate). The headline features lenders publish (grade,
home ownership, verification status, interest rate) are flat against
default in this dataset — max absolute correlation is 0.011. The
question we set out to answer: can a model find any signal at all,
and does Random Forest do better than Logistic Regression?

Outputs:
  - assets/*.png         Charts referenced inline in the blog HTML
  - output/loan_default_metrics.json  Numbers the HTML template substitutes in

Mirrors the structure of credit_scoring_breakdown.py — same theme
flags, same chart save helper, same output layout.
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
    confusion_matrix,
    f1_score,
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
DATA_PATH = ROOT / "datasets" / "train.csv"

parser = argparse.ArgumentParser(description="Loan default prediction blog analysis")
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
# Load and clean
# ---------------------------------------------------------------------------

print("Loading loan default dataset...")
df = pd.read_csv(DATA_PATH)
print(f"  rows={len(df):,}  cols={df.shape[1]}")

# Rename the obviously mislabeled column — its values are MORTGAGE/OWN/RENT,
# which is home ownership, not employment duration. The numeric column
# already named "Home Ownership" actually contains dollar amounts (likely
# borrower-reported home value); rename it for clarity.
df = df.rename(columns={
    "Employment Duration": "Home Ownership Type",
    "Home Ownership": "Home Value Reported",
})

# Drop the dead-weight columns.
DROP_COLS = [
    "ID",
    "Payment Plan",          # constant ("n" for every row)
    "Batch Enrolled",        # ~5k unique batch IDs; pure identifier noise
    "Loan Title",            # case-inconsistent free text; "Loan Intent"-ish but messy
    "Sub Grade",             # collinear with Grade
    "Initial List Status",   # near-constant; not predictive
    "Accounts Delinquent",   # constant 0 in this slice
]
df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
print(f"  after drop: {df.shape}")

TARGET = "Loan Status"

# Headline numbers
n_total = int(len(df))
n_default = int(df[TARGET].sum())
default_rate = float(df[TARGET].mean())
print(f"  defaults={n_default:,}  default_rate={default_rate:.2%}")

# ---------------------------------------------------------------------------
# Chart 1: Class imbalance
# ---------------------------------------------------------------------------

print("Building chart 1 — class imbalance...")
fig, ax = plt.subplots(figsize=(7, 4.2))
counts = df[TARGET].value_counts().sort_index()
labels = [f"Repaid\n{counts[0]:,}  ({(1-default_rate)*100:.1f}%)",
          f"Defaulted\n{counts[1]:,}  ({default_rate*100:.1f}%)"]
colors = [ACCENT, DANGER]
bars = ax.barh(["Repaid", "Defaulted"], counts.values, color=colors,
               edgecolor=PALETTE["spine"], linewidth=0.6)
for bar, label in zip(bars, labels):
    ax.text(bar.get_width() + counts.max() * 0.01,
            bar.get_y() + bar.get_height() / 2,
            label, va="center", fontsize=10, color=INK)
ax.set_title("Class imbalance: only 9.25% of loans defaulted")
ax.set_xlabel("Loan count")
ax.set_xlim(0, counts.max() * 1.25)
ax.invert_yaxis()
save_fig(fig, "loan_class_imbalance.png")

# ---------------------------------------------------------------------------
# Chart 2: The flat features — Grade, Home Ownership Type, Verification
# ---------------------------------------------------------------------------

print("Building chart 2 — the flat headline features...")
grade_order = ["A", "B", "C", "D", "E", "F", "G"]
by_grade = df.groupby("Grade")[TARGET].agg(["mean", "count"]).reindex(grade_order)
by_home = df.groupby("Home Ownership Type")[TARGET].agg(["mean", "count"])
by_verif = df.groupby("Verification Status")[TARGET].agg(["mean", "count"])

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
y_max = max(by_grade["mean"].max(), by_home["mean"].max(), by_verif["mean"].max()) * 100 + 5

axes[0].bar(by_grade.index, by_grade["mean"] * 100, color=ACCENT,
            edgecolor=PALETTE["spine"], linewidth=0.6)
axes[0].set_title("Loan grade")
axes[0].set_ylabel("Default rate (%)")
axes[0].set_ylim(0, y_max)
axes[0].axhline(default_rate * 100, color=MUTED, linestyle="--", linewidth=1, label=f"Overall {default_rate*100:.1f}%")
axes[0].legend(frameon=False, fontsize=8, loc="upper left")
axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))

axes[1].bar(by_home.index, by_home["mean"] * 100, color=GOLD,
            edgecolor=PALETTE["spine"], linewidth=0.6)
axes[1].set_title("Home ownership")
axes[1].set_ylim(0, y_max)
axes[1].axhline(default_rate * 100, color=MUTED, linestyle="--", linewidth=1)
axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))

axes[2].bar(by_verif.index, by_verif["mean"] * 100, color=PURPLE,
            edgecolor=PALETTE["spine"], linewidth=0.6)
axes[2].set_title("Verification status")
axes[2].set_ylim(0, y_max)
axes[2].axhline(default_rate * 100, color=MUTED, linestyle="--", linewidth=1)
axes[2].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
for label in axes[2].get_xticklabels():
    label.set_rotation(15)
    label.set_ha("right")

fig.suptitle("The headline features lenders advertise are nearly flat against default",
             color=NAVY, fontweight="bold", fontsize=14, y=1.02)
fig.tight_layout()
save_fig(fig, "loan_flat_features.png")

# ---------------------------------------------------------------------------
# Chart 3: Where signal actually lives — Public Record + Delinquency
# ---------------------------------------------------------------------------

print("Building chart 3 — signal-bearing features (Public Record, Delinquency)...")
# Bucket sparse high-count tails so the small-n outliers don't dominate.
df["_pr_bin"] = pd.cut(
    df["Public Record"],
    bins=[-0.5, 0.5, 1.5, 2.5, 100],
    labels=["0", "1", "2", "3+"],
)
df["_del_bin"] = pd.cut(
    df["Delinquency - two years"],
    bins=[-0.5, 0.5, 1.5, 3.5, 100],
    labels=["0", "1", "2-3", "4+"],
)
by_pr = df.groupby("_pr_bin", observed=True)[TARGET].agg(["mean", "count"])
by_del = df.groupby("_del_bin", observed=True)[TARGET].agg(["mean", "count"])

fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

bars1 = axes[0].bar(by_pr.index.astype(str), by_pr["mean"] * 100, color=DANGER,
                    edgecolor=PALETTE["spine"], linewidth=0.6)
for bar, n in zip(bars1, by_pr["count"]):
    h = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width() / 2, h + 0.2,
                 f"{h:.1f}%\nn={n:,}", ha="center", va="bottom", fontsize=9, color=INK)
axes[0].set_title("Public records on file")
axes[0].set_xlabel("Number of public records")
axes[0].set_ylabel("Default rate (%)")
axes[0].axhline(default_rate * 100, color=MUTED, linestyle="--", linewidth=1, label=f"Overall {default_rate*100:.1f}%")
axes[0].set_ylim(0, by_pr["mean"].max() * 100 + 3)
axes[0].legend(frameon=False, fontsize=9, loc="upper left")
axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))

bars2 = axes[1].bar(by_del.index.astype(str), by_del["mean"] * 100, color=DANGER,
                    edgecolor=PALETTE["spine"], linewidth=0.6)
for bar, n in zip(bars2, by_del["count"]):
    h = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width() / 2, h + 0.2,
                 f"{h:.1f}%\nn={n:,}", ha="center", va="bottom", fontsize=9, color=INK)
axes[1].set_title("Delinquencies in the past two years")
axes[1].set_xlabel("Count of 2-year delinquencies")
axes[1].axhline(default_rate * 100, color=MUTED, linestyle="--", linewidth=1)
axes[1].set_ylim(0, by_del["mean"].max() * 100 + 3)
axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))

fig.suptitle("The features that DO carry signal are negative-history flags",
             color=NAVY, fontweight="bold", fontsize=14, y=1.02)
fig.tight_layout()
save_fig(fig, "loan_signal_features.png")

# ---------------------------------------------------------------------------
# Chart 4: Interest rate distributions overlap (visual proof of the flat finding)
# ---------------------------------------------------------------------------

print("Building chart 4 — interest rate distribution overlap...")
fig, ax = plt.subplots(figsize=(8.5, 4.5))
rate_repaid = df.loc[df[TARGET] == 0, "Interest Rate"]
rate_default = df.loc[df[TARGET] == 1, "Interest Rate"]
sns.kdeplot(rate_repaid, ax=ax, color=ACCENT, fill=True, alpha=0.28,
            label=f"Repaid (mean {rate_repaid.mean():.2f}%)", linewidth=2)
sns.kdeplot(rate_default, ax=ax, color=DANGER, fill=True, alpha=0.28,
            label=f"Defaulted (mean {rate_default.mean():.2f}%)", linewidth=2)
ax.set_title("Interest rate distributions are statistically indistinguishable")
ax.set_xlabel("Interest rate (%)")
ax.set_ylabel("Density")
ax.set_xlim(rate_repaid.min(), rate_repaid.quantile(0.995))
ax.legend(frameon=False)
save_fig(fig, "loan_rate_overlap.png")

# ---------------------------------------------------------------------------
# Modeling: prepare feature matrix
# ---------------------------------------------------------------------------

print("Preparing feature matrix...")
df_model = df.drop(columns=["_pr_bin", "_del_bin"])

# Convert object/string categoricals to one-hot
cat_cols = df_model.select_dtypes(include=["object", "string"]).columns.tolist()
print(f"  one-hot encoding: {cat_cols}")
X = pd.get_dummies(df_model.drop(columns=[TARGET]), columns=cat_cols, drop_first=True).astype(float)
y = df_model[TARGET].astype(int).values
print(f"  X shape: {X.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
print(f"  train={len(X_train):,}  test={len(X_test):,}")

# ---------------------------------------------------------------------------
# Models — both trained without class weighting, then evaluated at a
# prevalence-matched threshold so each model predicts ~9.25% of test
# loans as defaults. Apples-to-apples at the same "lender risk appetite."
# ---------------------------------------------------------------------------

def threshold_at_predicted_rate(probs: np.ndarray, target_rate: float) -> float:
    """The probability cutoff that flags exactly `target_rate` of the test set."""
    return float(np.quantile(probs, 1 - target_rate))


def evaluate(name: str, probs: np.ndarray, y_true: np.ndarray, thresh: float) -> dict:
    preds = (probs >= thresh).astype(int)
    cm = confusion_matrix(y_true, preds)
    metrics = {
        "auc": float(roc_auc_score(y_true, probs)),
        "accuracy": float((preds == y_true).mean()),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "threshold": thresh,
        "predicted_default_rate": float(preds.mean()),
        "cm": cm,
    }
    print(f"  {name:24} auc={metrics['auc']:.3f}  acc={metrics['accuracy']:.3f}  "
          f"prec={metrics['precision']:.3f}  rec={metrics['recall']:.3f}  "
          f"f1={metrics['f1']:.3f}  thresh={thresh:.4f}")
    return metrics


print("Training logistic regression...")
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

lr = LogisticRegression(max_iter=4000, C=1.0)
lr.fit(X_train_s, y_train)
lr_probs = lr.predict_proba(X_test_s)[:, 1]

print("Training random forest (this takes ~30s)...")
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=4,
    n_jobs=-1,
    random_state=42,
)
rf.fit(X_train, y_train)
rf_probs = rf.predict_proba(X_test)[:, 1]

# Pick per-model thresholds that flag the bottom 9.25% as predicted-default.
lr_thresh = threshold_at_predicted_rate(lr_probs, default_rate)
rf_thresh = threshold_at_predicted_rate(rf_probs, default_rate)

print(f"Evaluating at prevalence-matched thresholds (target {default_rate:.4f}):")
lr_eval = evaluate("Logistic regression", lr_probs, y_test, lr_thresh)
rf_eval = evaluate("Random forest", rf_probs, y_test, rf_thresh)

lr_auc, lr_acc = lr_eval["auc"], lr_eval["accuracy"]
lr_prec, lr_rec, lr_f1 = lr_eval["precision"], lr_eval["recall"], lr_eval["f1"]
lr_cm = lr_eval["cm"]

rf_auc, rf_acc = rf_eval["auc"], rf_eval["accuracy"]
rf_prec, rf_rec, rf_f1 = rf_eval["precision"], rf_eval["recall"], rf_eval["f1"]
rf_cm = rf_eval["cm"]

# ---------------------------------------------------------------------------
# Chart 5: Model comparison metric bars
# ---------------------------------------------------------------------------

print("Building chart 5 — model metric comparison...")
metric_names = ["ROC AUC", "Precision", "Recall", "F1"]
lr_vals = [lr_auc, lr_prec, lr_rec, lr_f1]
rf_vals = [rf_auc, rf_prec, rf_rec, rf_f1]
x = np.arange(len(metric_names))
width = 0.38

fig, ax = plt.subplots(figsize=(8.5, 4.6))
b1 = ax.bar(x - width / 2, lr_vals, width, label="Logistic regression",
            color=ACCENT, edgecolor=PALETTE["spine"], linewidth=0.6)
b2 = ax.bar(x + width / 2, rf_vals, width, label="Random forest",
            color=PURPLE, edgecolor=PALETTE["spine"], linewidth=0.6)

for bars, vals in [(b1, lr_vals), (b2, rf_vals)]:
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9, color=INK)

ax.set_xticks(x)
ax.set_xticklabels(metric_names)
ax.set_ylim(0, max(max(lr_vals), max(rf_vals)) * 1.18)
ax.set_ylabel("Score")
ax.set_title("Random forest beats logistic regression — but both are modest")
ax.legend(frameon=False, loc="upper right")
save_fig(fig, "loan_model_comparison.png")

# ---------------------------------------------------------------------------
# Chart 6: ROC curves overlaid
# ---------------------------------------------------------------------------

print("Building chart 6 — ROC overlay...")
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probs)
fig, ax = plt.subplots(figsize=(6.8, 5.6))
ax.plot(lr_fpr, lr_tpr, color=ACCENT, linewidth=2.5,
        label=f"Logistic regression (AUC = {lr_auc:.3f})")
ax.plot(rf_fpr, rf_tpr, color=PURPLE, linewidth=2.5,
        label=f"Random forest (AUC = {rf_auc:.3f})")
ax.plot([0, 1], [0, 1], color=MUTED, linestyle="--", linewidth=1.2,
        label="Random baseline (AUC = 0.5)")
ax.set_title("ROC curves: random forest captures nonlinear signal LR misses")
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.02)
ax.legend(frameon=False, loc="lower right")
save_fig(fig, "loan_roc_overlay.png")

# ---------------------------------------------------------------------------
# Chart 7: Confusion matrices side-by-side
# ---------------------------------------------------------------------------

print("Building chart 7 — confusion matrices side-by-side...")
cm_cmap = "mako" if args.theme == "dark" else "Blues"
cm_annot_color = "#F8FAFC" if args.theme == "dark" else NAVY

fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
for ax_i, (cm, title) in enumerate([(lr_cm, "Logistic regression"), (rf_cm, "Random forest")]):
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
        xticklabels=["Predicted repay", "Predicted default"],
        yticklabels=["Actual repay", "Actual default"],
        ax=axes[ax_i],
        annot_kws={"fontsize": 11, "color": cm_annot_color, "weight": "bold"},
    )
    axes[ax_i].set_title(title)

fig.suptitle(
    f"Confusion matrices — each model's threshold tuned to flag the top {default_rate*100:.1f}% as predicted defaults",
    color=NAVY, fontweight="bold", fontsize=13, y=1.02)
fig.tight_layout()
save_fig(fig, "loan_confusion_matrices.png")

# ---------------------------------------------------------------------------
# Chart 8: Feature importance (random forest)
# ---------------------------------------------------------------------------

print("Building chart 8 — RF feature importance...")
fi = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=True)
top = fi.tail(14)
fig, ax = plt.subplots(figsize=(9, 6.5))
bars = ax.barh(top.index, top.values, color=PURPLE,
               edgecolor=PALETTE["spine"], linewidth=0.6)
for bar, v in zip(bars, top.values):
    ax.text(v + top.max() * 0.012, bar.get_y() + bar.get_height() / 2,
            f"{v:.3f}", va="center", fontsize=9, color=INK)
ax.set_title("Random forest feature importance — top 14")
ax.set_xlabel("Mean decrease in impurity")
ax.set_xlim(0, top.max() * 1.18)
save_fig(fig, "loan_feature_importance.png")

# ---------------------------------------------------------------------------
# Metrics export
# ---------------------------------------------------------------------------

print("Writing metrics.json...")
metrics = {
    "n_total": n_total,
    "n_default": n_default,
    "default_rate_pct": round(default_rate * 100, 2),
    "n_test": int(len(y_test)),
    "n_features": int(X.shape[1]),
    "flat_features": {
        "grade_min_pct": round(by_grade["mean"].min() * 100, 2),
        "grade_max_pct": round(by_grade["mean"].max() * 100, 2),
        "grade_spread_pp": round((by_grade["mean"].max() - by_grade["mean"].min()) * 100, 2),
        "home_ownership_min_pct": round(by_home["mean"].min() * 100, 2),
        "home_ownership_max_pct": round(by_home["mean"].max() * 100, 2),
        "verification_min_pct": round(by_verif["mean"].min() * 100, 2),
        "verification_max_pct": round(by_verif["mean"].max() * 100, 2),
        "rate_mean_repaid": round(float(rate_repaid.mean()), 2),
        "rate_mean_default": round(float(rate_default.mean()), 2),
    },
    "signal_features": {
        "public_record_0_pct": round(float(by_pr.loc["0", "mean"]) * 100, 2),
        "public_record_3plus_pct": round(float(by_pr.loc["3+", "mean"]) * 100, 2),
        "delinquency_0_pct": round(float(by_del.loc["0", "mean"]) * 100, 2),
        "delinquency_4plus_pct": round(float(by_del.loc["4+", "mean"]) * 100, 2),
    },
    "models": {
        "logistic_regression": {
            "auc": round(lr_auc, 3),
            "accuracy": round(lr_acc, 3),
            "precision": round(lr_prec, 3),
            "recall": round(lr_rec, 3),
            "f1": round(lr_f1, 3),
            "threshold": round(lr_thresh, 4),
            "confusion": {
                "tn": int(lr_cm[0, 0]), "fp": int(lr_cm[0, 1]),
                "fn": int(lr_cm[1, 0]), "tp": int(lr_cm[1, 1]),
            },
        },
        "random_forest": {
            "auc": round(rf_auc, 3),
            "accuracy": round(rf_acc, 3),
            "precision": round(rf_prec, 3),
            "recall": round(rf_rec, 3),
            "f1": round(rf_f1, 3),
            "threshold": round(rf_thresh, 4),
            "confusion": {
                "tn": int(rf_cm[0, 0]), "fp": int(rf_cm[0, 1]),
                "fn": int(rf_cm[1, 0]), "tp": int(rf_cm[1, 1]),
            },
        },
        "auc_uplift_pct": round((rf_auc - lr_auc) * 100, 2),
        "rf_recall_uplift_vs_lr_pct": round((rf_rec - lr_rec) * 100, 2),
    },
    "rf_top_features": [
        {"feature": k, "importance": round(float(v), 4)} for k, v in fi.tail(10)[::-1].items()
    ],
}
(OUTPUT / "loan_default_metrics.json").write_text(json.dumps(metrics, indent=2))
print(f"\nMetrics written to {(OUTPUT / 'loan_default_metrics.json').relative_to(ROOT)}")
print("Done.")
