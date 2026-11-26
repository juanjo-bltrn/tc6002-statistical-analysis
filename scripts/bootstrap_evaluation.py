import os
import re
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Figures are saved, not shown
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

OUTPUT_DIR = os.path.join("figures", "bootstrap-evaluation")

# ---------------------------------------------------------------------------
# Target levels (the target is ALREADY grouped into four categories)
# ---------------------------------------------------------------------------
# Edit these labels if your grouped column uses different spellings.
LEVELS = ["Insufficient_Weight", "Normal_Weight", "Overweight", "Obesity"]
COLORS = {
    "Insufficient": "#1f77b4",
    "Normal": "#2ca02c",
    "Overweight": "#ff7f0e",
    "Obesity": "#d62728",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _safe_name(name: str) -> str:
    """File-system friendly version of a column name."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name))


def _fmt_p(p: float) -> str:
    """Format a p-value for plot annotations."""
    if pd.isna(p):
        return "p = NA"
    if p < 1e-4:
        return "p < 1e-4"
    return f"p = {p:.4f}"


def prepare_target(df: pd.DataFrame, target: str = "NObeyesdad") -> pd.DataFrame:
    """Validate and order the four-level obesity target column."""
    data = df.copy()
    if target not in data.columns:
        raise KeyError(f"Target column '{target}' not found in dataframe.")
    present = [lvl for lvl in LEVELS if lvl in set(data[target].dropna())]
    if len(present) < 2:
        raise ValueError(
            f"Expected grouped target levels {LEVELS} in '{target}', "
            f"found {sorted(data[target].dropna().unique().tolist())}."
        )
    data[target] = pd.Categorical(data[target], categories=present, ordered=True)
    return data


def bh_adjust(p_values: pd.Series) -> np.ndarray:
    """Benjamini-Hochberg FDR adjustment."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return np.array([], dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adj = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        val = min(prev, ranked[i] * n / (i + 1))
        adj[i] = val
        prev = val
    out = np.empty(n, dtype=float)
    out[order] = np.clip(adj, 0.0, 1.0)
    return out


# ---------------------------------------------------------------------------
# 1. Discriminative Intervals & Kolmogorov-Smirnov Tests
# ---------------------------------------------------------------------------
def identify_discriminative_intervals(
    data: pd.DataFrame,
    feature: str,
    target: str = "NObeyesdad",
    n_grid: int = 1000,
    min_mass: float = 0.01,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, np.ndarray]]:
    """
    For a continuous feature:
      - Run one-vs-rest two-sample Kolmogorov-Smirnov (KS) tests for each level.
      - Evaluate Scott's-rule Gaussian KDEs on a shared grid.
      - Identify contiguous discriminative intervals (a, b] where level g's
        conditional PDF strictly dominates all other levels' conditional PDFs:
            f(x | Y = g) > max_{k != g} f(x | Y = k).

    Returns (ks_df, intervals_df, kde_curves).
    """
    sub = data[[feature, target]].dropna()
    x_all = sub[feature].to_numpy(dtype=float)
    grid = np.linspace(x_all.min(), x_all.max(), n_grid)

    levels = [lvl for lvl in data[target].cat.categories if (sub[target] == lvl).sum() >= 2]
    pdfs: Dict[str, np.ndarray] = {}
    ks_rows = []

    for lvl in levels:
        x_pos = sub.loc[sub[target] == lvl, feature].to_numpy(dtype=float)
        x_neg = sub.loc[sub[target] != lvl, feature].to_numpy(dtype=float)

        ks_stat, ks_p = stats.ks_2samp(x_pos, x_neg, alternative="two-sided")
        ks_rows.append(
            {
                "feature": feature,
                "level": lvl,
                "n_level": len(x_pos),
                "n_rest": len(x_neg),
                "ks_stat": float(ks_stat),
                "ks_p": float(ks_p),
            }
        )

        # Gaussian KDE (Scott's rule); fallback to tiny jitter if constant
        if np.std(x_pos, ddof=1) == 0:
            pdfs[lvl] = np.zeros_like(grid)
        else:
            kde = stats.gaussian_kde(x_pos, bw_method="scott")
            pdfs[lvl] = kde(grid)

    # Identify contiguous intervals (a, b] where each level has the highest PDF
    pdf_matrix = np.vstack([pdfs[lvl] for lvl in levels])
    winner_idx = np.argmax(pdf_matrix, axis=0)

    interval_rows = []
    for idx, lvl in enumerate(levels):
        mask = winner_idx == idx
        if not np.any(mask):
            continue

        # Find contiguous True runs
        padded = np.concatenate(([False], mask, [False]))
        starts = np.where(~padded[:-1] & padded[1:])[0]
        ends = np.where(padded[:-1] & ~padded[1:])[0] - 1

        # Compute max density of competing levels for ratio/margin reporting
        other_max = np.max(np.delete(pdf_matrix, idx, axis=0), axis=0)

        for s, e in zip(starts, ends):
            a, b = float(grid[s]), float(grid[e])
            seg_grid = grid[s : e + 1]
            seg_pdf = pdfs[lvl][s : e + 1]
            prob_mass = float(np.trapezoid(seg_pdf, seg_grid)) if e > s else 0.0
            if prob_mass < min_mass:
                continue

            peak_diff = float(np.max(seg_pdf - other_max[s : e + 1]))
            interval_rows.append(
                {
                    "feature": feature,
                    "level": lvl,
                    "interval_a": a,
                    "interval_b": b,
                    "interval_str": f"({a:.3f}, {b:.3f}]",
                    "width": b - a,
                    "level_prob_mass": prob_mass,
                    "max_pdf_margin": peak_diff,
                }
            )

    kde_curves = {"grid": grid, "pdfs": pdfs}
    return pd.DataFrame(ks_rows), pd.DataFrame(interval_rows), kde_curves


# ---------------------------------------------------------------------------
# 2. ROC, AUC, Youden's J, Confusion Matrix & Bootstrap Evaluation
# ---------------------------------------------------------------------------
def _compute_roc(y_true: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Compute ROC curve (FPR, TPR, thresholds) and AUC in pure NumPy.
    Assumes higher `scores` indicate the positive class (y_true == 1).
    """
    order = np.argsort(scores)[::-1]
    scores_sorted = scores[order]
    y_sorted = y_true[order]

    # Indices where score changes (to handle ties cleanly)
    distinct = np.where(np.diff(scores_sorted))[0]
    thresh_idx = np.r_[distinct, y_sorted.size - 1]

    tps = np.cumsum(y_sorted)[thresh_idx]
    fps = (1 + thresh_idx) - tps

    n_pos = y_sorted.sum()
    n_neg = y_sorted.size - n_pos

    tpr = np.r_[0.0, tps / n_pos]
    fpr = np.r_[0.0, fps / n_neg]
    thresholds = np.r_[np.inf, scores_sorted[thresh_idx]]
    auc = float(np.trapezoid(tpr, fpr))
    return fpr, tpr, thresholds, auc


def _evaluate_binary_classifier(
    x: np.ndarray,
    y_true: np.ndarray,
    direction: Optional[str] = None,
    fixed_threshold: Optional[float] = None,
) -> Dict[str, object]:
    """
    Evaluate a single continuous feature as a one-vs-rest binary classifier.
    Automatically selects orientation ('>=' vs '<=') so AUC >= 0.5 unless
    `direction` is provided. Uses Youden's J = TPR - FPR to select the
    optimal threshold unless `fixed_threshold` is provided.
    """
    if direction is None:
        fpr_hi, tpr_hi, thr_hi, auc_hi = _compute_roc(y_true, x)
        if auc_hi >= 0.5:
            direction = ">="
            fpr, tpr, thresholds, auc = fpr_hi, tpr_hi, thr_hi, auc_hi
        else:
            direction = "<="
            fpr, tpr, thr_neg, auc = _compute_roc(y_true, -x)
            thresholds = -thr_neg
    else:
        scores = x if direction == ">=" else -x
        fpr, tpr, thr_raw, auc = _compute_roc(y_true, scores)
        thresholds = thr_raw if direction == ">=" else -thr_raw

    if fixed_threshold is None:
        youden_j = tpr - fpr
        best_idx = int(np.argmax(youden_j))
        threshold = float(thresholds[best_idx])
        best_j = float(youden_j[best_idx])
    else:
        threshold = float(fixed_threshold)
        best_j = np.nan

    y_pred = (x >= threshold) if direction == ">=" else (x <= threshold)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    type1_error = fp / (fp + tn) if (fp + tn) > 0 else np.nan  # FPR (alpha)
    type2_error = fn / (fn + tp) if (fn + tp) > 0 else np.nan  # FNR (beta)
    power = 1.0 - type2_error if not np.isnan(type2_error) else np.nan  # TPR / Sensitivity
    specificity = 1.0 - type1_error if not np.isnan(type1_error) else np.nan

    return {
        "direction": direction,
        "threshold": threshold,
        "youden_j": best_j if not np.isnan(best_j) else (power - type1_error),
        "auc": auc,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "type1_error": float(type1_error),
        "type2_error": float(type2_error),
        "power": float(power),
        "specificity": float(specificity),
        "fpr_curve": fpr,
        "tpr_curve": tpr,
    }


def bootstrap_evaluate_feature(
    data: pd.DataFrame,
    feature: str,
    target: str = "NObeyesdad",
    n_boot: int = 1000,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, np.ndarray]]]:
    """
    Evaluate `feature` as a one-vs-rest binary classifier for each obesity level
    using bootstrap resampling (n_boot iterations) to obtain 95% CIs for AUC,
    Type I error (alpha), Type II error (beta), and statistical power (1 - beta).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    sub = data[[feature, target]].dropna()
    x_all = sub[feature].to_numpy(dtype=float)
    y_cat = sub[target].to_numpy()
    n = len(x_all)

    levels = [lvl for lvl in data[target].cat.categories if np.sum(y_cat == lvl) >= 2]
    metric_rows, cm_rows = [], []
    roc_curves: Dict[str, Dict[str, np.ndarray]] = {}

    for lvl in levels:
        y_bin = (y_cat == lvl).astype(int)
        obs = _evaluate_binary_classifier(x_all, y_bin)

        # Bootstrap resampling (evaluating AUC and the fixed Youden threshold)
        boot_auc = np.empty(n_boot, dtype=float)
        boot_t1 = np.empty(n_boot, dtype=float)
        boot_t2 = np.empty(n_boot, dtype=float)
        boot_pow = np.empty(n_boot, dtype=float)

        for b in range(n_boot):
            idx = rng.integers(0, n, size=n)
            yb = y_bin[idx]
            # Redraw if a bootstrap sample happens to be single-class
            while yb.sum() == 0 or yb.sum() == n:
                idx = rng.integers(0, n, size=n)
                yb = y_bin[idx]
            xb = x_all[idx]

            res_b = _evaluate_binary_classifier(
                xb,
                yb,
                direction=obs["direction"],
                fixed_threshold=obs["threshold"],
            )
            boot_auc[b] = res_b["auc"]
            boot_t1[b] = res_b["type1_error"]
            boot_t2[b] = res_b["type2_error"]
            boot_pow[b] = res_b["power"]

        auc_lo, auc_hi = np.percentile(boot_auc, [2.5, 97.5])
        t1_lo, t1_hi = np.percentile(boot_t1, [2.5, 97.5])
        t2_lo, t2_hi = np.percentile(boot_t2, [2.5, 97.5])
        pow_lo, pow_hi = np.percentile(boot_pow, [2.5, 97.5])

        metric_rows.append(
            {
                "feature": feature,
                "level": lvl,
                "rule": f"{feature} {obs['direction']} {obs['threshold']:.4f}",
                "direction": obs["direction"],
                "threshold": obs["threshold"],
                "youden_j": obs["youden_j"],
                "auc": obs["auc"],
                "auc_ci_low": float(auc_lo),
                "auc_ci_high": float(auc_hi),
                "type1_error": obs["type1_error"],
                "type1_ci_low": float(t1_lo),
                "type1_ci_high": float(t1_hi),
                "type2_error": obs["type2_error"],
                "type2_ci_low": float(t2_lo),
                "type2_ci_high": float(t2_hi),
                "power": obs["power"],
                "power_ci_low": float(pow_lo),
                "power_ci_high": float(pow_hi),
                "specificity": obs["specificity"],
            }
        )

        cm_rows.append(
            {
                "feature": feature,
                "level": lvl,
                "threshold": obs["threshold"],
                "direction": obs["direction"],
                "TP": obs["tp"],
                "FP": obs["fp"],
                "TN": obs["tn"],
                "FN": obs["fn"],
            }
        )

        roc_curves[lvl] = {
            "fpr": obs["fpr_curve"],
            "tpr": obs["tpr_curve"],
            "auc": obs["auc"],
            "auc_ci_low": float(auc_lo),
            "auc_ci_high": float(auc_hi),
            "opt_fpr": obs["type1_error"],
            "opt_tpr": obs["power"],
        }

    return pd.DataFrame(metric_rows), pd.DataFrame(cm_rows), roc_curves


def summarize_best_features(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify which feature achieves:
      - Highest statistical power (1 - Type II error) per level and overall.
      - Lowest observed Type I error (alpha) per level and overall.
    """
    rows = []

    # Per-level winners
    for lvl, grp in metrics_df.groupby("level", sort=False):
        best_pow = grp.loc[grp["power"].idxmax()]
        best_t1 = grp.loc[grp["type1_error"].idxmin()]
        rows.append(
            {
                "scope": f"Level: {lvl}",
                "highest_power_feature": best_pow["feature"],
                "highest_power_value": float(best_pow["power"]),
                "lowest_type1_feature": best_t1["feature"],
                "lowest_type1_value": float(best_t1["type1_error"]),
            }
        )

    # Overall across all levels (mean across levels)
    agg = (
        metrics_df.groupby("feature", as_index=False)
        .agg(mean_power=("power", "mean"), mean_type1=("type1_error", "mean"), mean_auc=("auc", "mean"))
    )
    overall_pow = agg.loc[agg["mean_power"].idxmax()]
    overall_t1 = agg.loc[agg["mean_type1"].idxmin()]
    rows.append(
        {
            "scope": "Overall (Mean Across Levels)",
            "highest_power_feature": overall_pow["feature"],
            "highest_power_value": float(overall_pow["mean_power"]),
            "lowest_type1_feature": overall_t1["feature"],
            "lowest_type1_value": float(overall_t1["mean_type1"]),
        }
    )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Plotting
# ---------------------------------------------------------------------------
def plot_discriminative_intervals(
    feature: str,
    kde_curves: Dict[str, object],
    intervals_df: pd.DataFrame,
    ks_df: pd.DataFrame,
    out_dir: str,
) -> str:
    """Plot overlaid conditional KDEs with shaded discriminative intervals (a, b]."""
    grid = kde_curves["grid"]
    pdfs = kde_curves["pdfs"]

    fig, ax = plt.subplots(figsize=(9, 5))
    ks_lookup = ks_df.set_index("level").to_dict(orient="index")

    for lvl, pdf in pdfs.items():
        color = COLORS.get(lvl, None)
        ks_info = ks_lookup.get(lvl, {})
        label = (
            f"{lvl} (KS={ks_info.get('ks_stat', np.nan):.2f}, "
            f"{_fmt_p(ks_info.get('ks_p_fdr', ks_info.get('ks_p', np.nan)))})"
        )
        ax.plot(grid, pdf, lw=2.0, color=color, label=label)

    # Shade dominant discriminative intervals
    if not intervals_df.empty:
        for _, row in intervals_df.iterrows():
            color = COLORS.get(row["level"], "gray")
            ax.axvspan(row["interval_a"], row["interval_b"], color=color, alpha=0.12)

    ax.set_title(f"Discriminative Intervals — {feature}", fontsize=12, fontweight="bold")
    ax.set_xlabel(feature)
    ax.set_ylabel("Conditional Density (KDE)")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(frameon=True, fontsize=8.5)
    fig.tight_layout()

    path = os.path.join(out_dir, f"discriminative_intervals_{_safe_name(feature)}.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_roc_curves(
    feature: str,
    roc_curves: Dict[str, Dict[str, np.ndarray]],
    out_dir: str,
) -> str:
    """Plot one-vs-rest ROC curves with bootstrap 95% CIs and Youden's J points."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1.0, alpha=0.6, label="Chance (AUC = 0.50)")

    for lvl, info in roc_curves.items():
        color = COLORS.get(lvl, None)
        label = (
            f"{lvl}: AUC = {info['auc']:.3f} "
            f"[{info['auc_ci_low']:.3f}, {info['auc_ci_high']:.3f}]"
        )
        ax.plot(info["fpr"], info["tpr"], lw=2.0, color=color, label=label)
        ax.scatter(
            [info["opt_fpr"]],
            [info["opt_tpr"]],
            color=color,
            edgecolor="black",
            s=45,
            zorder=5,
        )

    ax.set_title(f"One-vs-Rest ROC Curves — {feature}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Type I Error (False Positive Rate, $\\alpha$)")
    ax.set_ylabel("Statistical Power (True Positive Rate, $1 - \\beta$)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(loc="lower right", frameon=True, fontsize=8.5)
    fig.tight_layout()

    path = os.path.join(out_dir, f"roc_curves_{_safe_name(feature)}.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_error_power_summary(metrics_df: pd.DataFrame, out_dir: str) -> str:
    """Side-by-side heatmaps of Statistical Power (1 - beta) and Type I Error (alpha)."""
    power_piv = metrics_df.pivot(index="feature", columns="level", values="power")
    t1_piv = metrics_df.pivot(index="feature", columns="level", values="type1_error")

    cols = [lvl for lvl in LEVELS if lvl in power_piv.columns]
    power_piv = power_piv[cols]
    t1_piv = t1_piv[cols]

    fig, axes = plt.subplots(1, 2, figsize=(12, max(4, 0.6 * len(power_piv) + 1.5)))
    panels = [
        (axes[0], power_piv, "Statistical Power ($1 - \\beta$)", "YlGn"),
        (axes[1], t1_piv, "Observed Type I Error ($\\alpha$)", "YlOrRd"),
    ]

    for ax, piv, title, cmap in panels:
        im = ax.imshow(piv.to_numpy(dtype=float), cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_xticks(range(len(piv.columns)))
        ax.set_xticklabels(piv.columns, rotation=25, ha="right")
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels(piv.index)
        ax.set_title(title, fontsize=11, fontweight="bold")

        for i in range(len(piv.index)):
            for j in range(len(piv.columns)):
                val = piv.iloc[i, j]
                if not pd.isna(val):
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center", color="black", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    path = os.path.join(out_dir, "power_and_type1_summary.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def analyze_discriminative_bootstrap(
    df: pd.DataFrame,
    continuous_features: List[str],
    target: str = "NObeyesdad",
    out_dir: str = OUTPUT_DIR,
    alpha: float = 0.05,
    n_boot: int = 1000,
    seed: int = 42,
) -> Dict[str, object]:
    """
    Run Task 6 (Discriminative Intervals and Bootstrap Evaluation) for all
    continuous features.

    Returns dict with DataFrames 'ks_tests', 'discriminative_intervals',
    'bootstrap_metrics', 'confusion_matrices', 'best_features', and the list
    of saved figure paths. CSV copies are written to `out_dir`.
    """
    missing = [c for c in continuous_features if c not in df.columns]
    if missing:
        raise KeyError(f"Columns not found in dataframe: {missing}")

    os.makedirs(out_dir, exist_ok=True)
    data = prepare_target(df, target)
    rng = np.random.default_rng(seed)

    ks_list, int_list, met_list, cm_list = [], [], [], []
    kde_ctx, roc_ctx = {}, {}

    for f in continuous_features:
        ks_f, int_f, kde_f = identify_discriminative_intervals(data, f, target=target)
        met_f, cm_f, roc_f = bootstrap_evaluate_feature(
            data, f, target=target, n_boot=n_boot, rng=rng
        )
        ks_list.append(ks_f)
        int_list.append(int_f)
        met_list.append(met_f)
        cm_list.append(cm_f)
        kde_ctx[f] = kde_f
        roc_ctx[f] = roc_f

    ks_df = pd.concat(ks_list, ignore_index=True)
    ks_df["ks_p_fdr"] = bh_adjust(ks_df["ks_p"])
    ks_df["significant_fdr"] = ks_df["ks_p_fdr"] < alpha

    intervals_df = pd.concat(int_list, ignore_index=True) if any(not d.empty for d in int_list) else pd.DataFrame()
    metrics_df = pd.concat(met_list, ignore_index=True)
    cm_df = pd.concat(cm_list, ignore_index=True)
    best_df = summarize_best_features(metrics_df)

    # Save CSV outputs
    ks_df.to_csv(os.path.join(out_dir, "ks_tests.csv"), index=False)
    intervals_df.to_csv(os.path.join(out_dir, "discriminative_intervals.csv"), index=False)
    metrics_df.to_csv(os.path.join(out_dir, "bootstrap_roc_metrics.csv"), index=False)
    cm_df.to_csv(os.path.join(out_dir, "confusion_matrices.csv"), index=False)
    best_df.to_csv(os.path.join(out_dir, "best_features_summary.csv"), index=False)

    # Generate figures
    figures = []
    for f in continuous_features:
        f_int = intervals_df[intervals_df["feature"] == f] if not intervals_df.empty else pd.DataFrame()
        f_ks = ks_df[ks_df["feature"] == f]
        figures.append(plot_discriminative_intervals(f, kde_ctx[f], f_int, f_ks, out_dir))
        figures.append(plot_roc_curves(f, roc_ctx[f], out_dir))
    figures.append(plot_error_power_summary(metrics_df, out_dir))

    return {
        "ks_tests": ks_df,
        "discriminative_intervals": intervals_df,
        "bootstrap_metrics": metrics_df,
        "confusion_matrices": cm_df,
        "best_features": best_df,
        "figures": figures,
    }