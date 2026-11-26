import os
import re
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import integrate, stats

OUTPUT_DIR = os.path.join("figures", "model-based-expectations")

# ---------------------------------------------------------------------------
#            Target levels & styling (consistent with Tasks 2–4)
# ---------------------------------------------------------------------------
LEVELS = [
    "Insufficient_Weight",
    "Normal_Weight",
    "Overweight",
    "Obesity",
]

COLORS = {
    "Insufficient_Weight": "#1f77b4",
    "Normal_Weight": "#2ca02c",
    "Overweight": "#ff7f0e",
    "Obesity": "#d62728",
}

TARGET_MAP = {
    "Insufficient_Weight": "Insufficient_Weight",
    "Normal_Weight": "Normal_Weight",
    "Overweight_Level_I": "Overweight",
    "Overweight_Level_II": "Overweight",
    "Obesity_Type_I": "Obesity",
    "Obesity_Type_II": "Obesity",
    "Obesity_Type_III": "Obesity",
}


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(name))


def prepare_target(
    df: pd.DataFrame, target: str = "NObeyesdad", level_col: str = "obesity_level"
) -> pd.DataFrame:
    """Map raw 7-class target to the 4 grouped obesity levels if not already mapped."""
    out = df.copy()
    if level_col not in out.columns:
        out[level_col] = out[target].replace(TARGET_MAP)
    return out


def _groups(df: pd.DataFrame, feature: str, level_col: str) -> Dict[str, np.ndarray]:
    return {
        lv: df.loc[df[level_col] == lv, feature].dropna().to_numpy(dtype=float)
        for lv in LEVELS
    }


# ---------------------------------------------------------------------------
# (a) Model-based conditional mean & variance from conditional PDF / KDE
# ---------------------------------------------------------------------------
def _extract_pdf_grid(
    x: np.ndarray,
    dist_entry: Optional[Dict[str, object]] = None,
    grid_size: int = 2000,
    pad_bw: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Retrieve (grid, pdf, bandwidth) from a Task 4 distribution entry, or fit a
    Gaussian KDE (Scott's rule) directly if Task 4 outputs are not supplied.

    If a fitted `kde` object is present in `dist_entry`, we evaluate it on an
    extended grid (+/- pad_bw * bandwidth beyond the data min/max) so that the
    Gaussian tails are fully captured during numerical integration.
    """
    kde = None
    if dist_entry is not None:
        kde = dist_entry.get("kde")
        if kde is None and "grid" in dist_entry and "pdf" in dist_entry:
            grid = np.asarray(dist_entry["grid"], dtype=float)
            pdf = np.asarray(dist_entry["pdf"], dtype=float)
            return grid, pdf, np.nan

    if kde is None:
        kde = stats.gaussian_kde(x, bw_method="scott")

    # Scott's bandwidth in data units: h = factor * sample_std
    bw = float(np.sqrt(kde.covariance.squeeze()))
    lo = float(x.min() - pad_bw * bw)
    hi = float(x.max() + pad_bw * bw)
    grid = np.linspace(lo, hi, grid_size)
    pdf = kde(grid)
    return grid, pdf, bw


def model_based_moments(
    grid: np.ndarray, pdf: np.ndarray, normalize: bool = True
) -> Dict[str, float]:
    """
    Compute model-based mean and variance from an estimated PDF via the
    trapezoidal rule:
        E[X]   = integral x * f(x) dx
        E[X^2] = integral x^2 * f(x) dx
        Var(X) = E[X^2] - (E[X])^2
    """
    area = float(integrate.trapezoid(pdf, grid))
    f = pdf / area if (normalize and area > 0) else pdf

    model_mean = float(integrate.trapezoid(grid * f, grid))
    second_moment = float(integrate.trapezoid((grid ** 2) * f, grid))
    model_var = float(max(second_moment - model_mean ** 2, 0.0))

    return {
        "pdf_integral": area,
        "model_mean": model_mean,
        "model_variance": model_var,
        "model_sd": float(np.sqrt(model_var)),
    }


def compare_expectations_feature(
    df: pd.DataFrame,
    feature: str,
    level_col: str = "obesity_level",
    task2_stats: Optional[pd.DataFrame] = None,
    task4_feature_dist: Optional[Dict[str, Dict[str, object]]] = None,
    tol: float = 1e-3,
    grid_size: int = 2000,
) -> pd.DataFrame:
    """
    For one continuous feature, compute model-based conditional mean & variance
    across all four obesity levels and compare against Task 2 sample statistics.
    """
    groups = _groups(df, feature, level_col)

    # Index Task 2 sample stats for fast lookup if provided
    t2_lookup = None
    if task2_stats is not None and not task2_stats.empty:
        sub = task2_stats.loc[task2_stats["feature"] == feature]
        if not sub.empty:
            t2_lookup = sub.set_index("level")

    rows = []
    for lv, x in groups.items():
        n = len(x)
        dist_entry = task4_feature_dist.get(lv) if task4_feature_dist else None
        grid, pdf, bw = _extract_pdf_grid(x, dist_entry=dist_entry, grid_size=grid_size)
        moments = model_based_moments(grid, pdf, normalize=True)

        # Retrieve sample mean & unbiased sample variance from Task 2 (or compute)
        if t2_lookup is not None and lv in t2_lookup.index:
            sample_mean = float(t2_lookup.loc[lv, "mean"])
            sample_var = float(t2_lookup.loc[lv, "variance"])
        else:
            sample_mean = float(x.mean())
            sample_var = float(x.var(ddof=1))

        mean_diff = moments["model_mean"] - sample_mean
        mean_abs_diff = abs(mean_diff)
        mean_rel_diff = mean_abs_diff / abs(sample_mean) if sample_mean != 0 else np.nan

        var_diff = moments["model_variance"] - sample_var
        var_abs_diff = abs(var_diff)
        var_rel_diff = var_abs_diff / abs(sample_var) if sample_var != 0 else np.nan

        # Theoretical Gaussian KDE variance: ((n - 1) / n) * s^2 + bw^2
        kde_var_theory = ((n - 1) / n) * sample_var + bw ** 2 if not np.isnan(bw) else np.nan

        rows.append({
            "feature": feature,
            "level": lv,
            "n": n,
            "kde_bandwidth": bw,
            "pdf_integral": moments["pdf_integral"],
            "sample_mean": sample_mean,
            "model_mean": moments["model_mean"],
            "mean_diff": mean_diff,
            "mean_abs_diff": mean_abs_diff,
            "mean_rel_diff_pct": 100.0 * mean_rel_diff,
            "mean_equivalent": mean_abs_diff <= tol,
            "sample_variance": sample_var,
            "model_variance": moments["model_variance"],
            "kde_variance_theory": kde_var_theory,
            "var_diff": var_diff,
            "var_abs_diff": var_abs_diff,
            "var_rel_diff_pct": 100.0 * var_rel_diff,
            "var_equivalent": var_abs_diff <= tol,
            "tolerance": tol,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_parity_comparison(res_df: pd.DataFrame, out_dir: str) -> str:
    """Two-panel parity plot: Model-Based vs. Sample Mean and Variance."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), dpi=150)

    for ax, s_col, m_col, title, xlabel, ylabel in [
        (
            axes[0],
            "sample_mean",
            "model_mean",
            "Conditional Mean: Model-Based vs. Sample",
            "Sample Conditional Mean (Task 2)",
            "Model-Based Conditional Mean (KDE)",
        ),
        (
            axes[1],
            "sample_variance",
            "model_variance",
            "Conditional Variance: Model-Based vs. Sample",
            "Sample Conditional Variance (Task 2)",
            "Model-Based Conditional Variance (KDE)",
        ),
    ]:
        for lv in LEVELS:
            sub = res_df.loc[res_df["level"] == lv]
            ax.scatter(
                sub[s_col],
                sub[m_col],
                label=lv.replace("_", " "),
                color=COLORS[lv],
                alpha=0.85,
                s=45,
                edgecolors="white",
                linewidths=0.5,
            )

        lims = [
            min(res_df[s_col].min(), res_df[m_col].min()),
            max(res_df[s_col].max(), res_df[m_col].max()),
        ]
        pad = 0.05 * (lims[1] - lims[0]) if lims[1] > lims[0] else 1.0
        ax.plot(
            [lims[0] - pad, lims[1] + pad],
            [lims[0] - pad, lims[1] + pad],
            linestyle="--",
            color="black",
            linewidth=1.0,
            label="y = x (exact equivalence)",
        )
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(fontsize=8, frameon=True)

    fig.tight_layout()
    path = os.path.join(out_dir, "model_vs_sample_parity.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_discrepancies(res_df: pd.DataFrame, tol: float, out_dir: str) -> str:
    """
    Grouped bar plot comparing absolute differences |Model - Sample| for
    conditional means (log scale) and relative % inflation for variances.
    """
    features = list(dict.fromkeys(res_df["feature"]))
    x = np.arange(len(features))
    width = 0.19

    fig, axes = plt.subplots(2, 1, figsize=(12, 8.5), dpi=150, sharex=True)

    # Panel 1: Absolute mean difference vs tolerance
    for i, lv in enumerate(LEVELS):
        sub = res_df.loc[res_df["level"] == lv].set_index("feature").reindex(features)
        vals = np.maximum(sub["mean_abs_diff"].to_numpy(), 1e-12)
        axes[0].bar(
            x + (i - 1.5) * width,
            vals,
            width=width,
            label=lv.replace("_", " "),
            color=COLORS[lv],
            alpha=0.88,
        )
    axes[0].axhline(tol, color="black", linestyle="--", linewidth=1.2, label=f"Tolerance ({tol:.0e})")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("|Model Mean - Sample Mean| (log)", fontsize=10)
    axes[0].set_title("Absolute Discrepancy in Conditional Means (KDE vs. Sample)", fontsize=11, fontweight="bold")
    axes[0].grid(True, axis="y", linestyle=":", alpha=0.5)
    axes[0].legend(ncol=5, fontsize=8, loc="upper right")

    # Panel 2: Relative percentage difference in variance
    for i, lv in enumerate(LEVELS):
        sub = res_df.loc[res_df["level"] == lv].set_index("feature").reindex(features)
        pct = 100.0 * (sub["var_diff"] / sub["sample_variance"]).to_numpy()
        axes[1].bar(
            x + (i - 1.5) * width,
            pct,
            width=width,
            label=lv.replace("_", " "),
            color=COLORS[lv],
            alpha=0.88,
        )
    axes[1].axhline(0.0, color="black", linewidth=0.9)
    axes[1].set_ylabel("Variance Difference (% of Sample Var)", fontsize=10)
    axes[1].set_title(
        "Relative Variance Inflation of Gaussian KDE vs. Sample Variance",
        fontsize=11,
        fontweight="bold",
    )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(features, rotation=25, ha="right", fontsize=9.5)
    axes[1].grid(True, axis="y", linestyle=":", alpha=0.5)

    fig.tight_layout()
    path = os.path.join(out_dir, "model_vs_sample_discrepancies.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def analyze_model_based_expectations(
    df: pd.DataFrame,
    continuous_features: List[str],
    target: str = "NObeyesdad",
    task2_stats: Optional[pd.DataFrame] = None,
    task4_distributions: Optional[Dict[str, Dict[str, Dict[str, object]]]] = None,
    tol: float = 1e-3,
    grid_size: int = 2000,
    out_dir: str = OUTPUT_DIR,
) -> Dict[str, object]:
    """
    Run Task 5 for all continuous features.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    continuous_features : List[str]
        Names of the continuous features.
    target : str
        Name of the raw target column (mapped to 4 levels if needed).
    task2_stats : pd.DataFrame, optional
        Output `conditional_stats` DataFrame from Task 2's
        `analyze_continuous_features()`. If None, sample mean and unbiased
        sample variance are computed directly from `df`.
    task4_distributions : dict, optional
        Output `continuous_distributions` dict from Task 4 (nested as
        `task4_distributions[feature][level] = {'grid', 'pdf', 'cdf', 'kde'}`).
        If None, Gaussian KDEs (Scott's rule) are fitted directly.
    tol : float
        Absolute equivalence tolerance (default: 1e-3).
    grid_size : int
        Number of grid points for trapezoidal numerical integration.
    out_dir : str
        Directory where CSV tables and PNG figures are saved.

    Returns
    -------
    dict with keys:
        - 'expectations': Full per-feature, per-level comparison DataFrame.
        - 'summary': Per-feature equivalence summary DataFrame.
        - 'figures': List of saved figure paths.
    """
    os.makedirs(out_dir, exist_ok=True)
    data = prepare_target(df, target)

    frames = []
    for f in continuous_features:
        t4_f = task4_distributions.get(f) if task4_distributions else None
        frames.append(
            compare_expectations_feature(
                data,
                feature=f,
                task2_stats=task2_stats,
                task4_feature_dist=t4_f,
                tol=tol,
                grid_size=grid_size,
            )
        )

    exp_df = pd.concat(frames, ignore_index=True)

    summary_df = (
        exp_df.groupby("feature", sort=False)
        .agg(
            n_levels=("level", "count"),
            mean_equiv_count=("mean_equivalent", "sum"),
            max_mean_abs_diff=("mean_abs_diff", "max"),
            var_equiv_count=("var_equivalent", "sum"),
            max_var_abs_diff=("var_abs_diff", "max"),
            mean_var_inflation_pct=("var_rel_diff_pct", "mean"),
        )
        .reset_index()
    )
    summary_df["all_means_equivalent"] = summary_df["mean_equiv_count"] == summary_df["n_levels"]
    summary_df["all_vars_equivalent"] = summary_df["var_equiv_count"] == summary_df["n_levels"]

    exp_df.to_csv(os.path.join(out_dir, "model_based_expectations.csv"), index=False)
    summary_df.to_csv(os.path.join(out_dir, "equivalence_summary.csv"), index=False)

    figures = [
        plot_parity_comparison(exp_df, out_dir),
        plot_discrepancies(exp_df, tol, out_dir),
    ]

    return {
        "expectations": exp_df,
        "summary": summary_df,
        "figures": figures,
    }


# ---------------------------------------------------------------------------
# Example pipeline usage (connecting Tasks 2, 4, and 5)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Example standalone or integrated call:
    # df = pd.read_csv("ObesityDataSet_raw_and_data_sinthetic.csv")
    # continuous_features = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]
    #
    # t2_out = analyze_continuous_features(df, continuous_features)
    # t4_out = analyze_conditional_distributions(df, continuous_features, ...)
    #
    # t5_out = analyze_model_based_expectations(
    #     df=df,
    #     continuous_features=continuous_features,
    #     task2_stats=t2_out["conditional_stats"],
    #     task4_distributions=t4_out["continuous_distributions"],
    #     tol=1e-3,
    # )
    # print(t5_out["summary"].to_string(index=False))
    pass