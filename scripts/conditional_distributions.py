import os
import re
from typing import Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")  # Figures are saved, not shown
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

OUTPUT_DIR = os.path.join("figures", "conditional-distributions")

DEFAULT_LEVEL_ORDER = [
    "Insufficient_Weight",
    "Normal_Weight",
    "Overweight",
    "Obesity",
]

LEVEL_COLORS = {
    "Insufficient_Weight": "#1f77b4",
    "Normal_Weight": "#2ca02c",
    "Overweight": "#ff7f0e",
    "Obesity": "#d62728",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _safe_name(name: str) -> str:
    """File-system friendly version of a column name."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name))


def _resolve_levels(series: pd.Series, level_order: Optional[Sequence] = None) -> List:
    """Return ordered list of obesity levels present in `series`."""
    present = list(series.dropna().unique())
    if level_order is not None:
        ordered = [lvl for lvl in level_order if lvl in present]
        extras = [lvl for lvl in present if lvl not in ordered]
        return ordered + extras
    return sorted(present, key=str)


def _resolve_categories(
    series: pd.Series, category_order: Optional[Sequence] = None
) -> List:
    """Return ordered list of categories present in `series`."""
    present = list(series.dropna().unique())
    if category_order is not None:
        ordered = [cat for cat in category_order if cat in present]
        extras = [cat for cat in present if cat not in ordered]
        return ordered + extras
    return sorted(present, key=str)


# ---------------------------------------------------------------------------
# Categorical & binary features: Conditional PMF + Conditional CDF
# ---------------------------------------------------------------------------
def conditional_categorical_pmf_cdf(
    df: pd.DataFrame,
    feature: str,
    target_col: str,
    level_order: Optional[Sequence] = None,
    category_order: Optional[Sequence] = None,
) -> pd.DataFrame:
    """
    Estimate the conditional PMF P(X = c | Y = g) and conditional CDF
    F(c | Y = g) = sum_{j <= c} P(X = j | Y = g) for a discrete feature.

    Returns a tidy DataFrame indexed by (obesity_level, category) with columns:
      count : observed frequency O_{gc}
      n_g   : stratum sample size
      pmf   : conditional empirical probability O_{gc} / n_g
      cdf   : conditional cumulative probability over `category_order`
    """
    sub = df[[feature, target_col]].dropna()
    levels = _resolve_levels(sub[target_col], level_order)
    categories = _resolve_categories(sub[feature], category_order)

    ct = pd.crosstab(sub[target_col], sub[feature]).reindex(
        index=levels, columns=categories, fill_value=0
    )

    records = []
    for lvl in levels:
        counts = ct.loc[lvl]
        n_g = int(counts.sum())
        pmf = counts / n_g if n_g > 0 else counts * 0.0
        cdf = pmf.cumsum()
        for cat in categories:
            records.append(
                {
                    "obesity_level": lvl,
                    "category": cat,
                    "count": int(counts[cat]),
                    "n_g": n_g,
                    "pmf": float(pmf[cat]),
                    "cdf": float(cdf[cat]),
                }
            )

    return pd.DataFrame(records).set_index(["obesity_level", "category"])


def plot_conditional_categorical(
    table: pd.DataFrame,
    feature: str,
    out_dir: str,
    level_order: Sequence,
    category_order: Sequence,
) -> str:
    """Plot conditional PMF (grouped bars) and conditional CDF (step curves)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(len(category_order))
    n_levels = len(level_order)
    bar_width = 0.8 / max(n_levels, 1)

    for idx, lvl in enumerate(level_order):
        sub = table.loc[lvl].reindex(category_order)
        color = LEVEL_COLORS.get(lvl, None)
        offset = (idx - (n_levels - 1) / 2.0) * bar_width

        # Left panel: Conditional PMF
        axes[0].bar(
            x + offset,
            sub["pmf"].to_numpy(),
            width=bar_width,
            label=str(lvl).replace("_", " "),
            color=color,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
        )

        # Right panel: Conditional CDF
        axes[1].step(
            x,
            sub["cdf"].to_numpy(),
            where="post",
            marker="o",
            markersize=4,
            linewidth=1.8,
            label=str(lvl).replace("_", " "),
            color=color,
        )

    cat_labels = [str(c).replace("_", " ") for c in category_order]
    for ax, ylabel, title in [
        (axes[0], "Conditional PMF  P(X = x | Y = g)", f"{feature}: Conditional PMF"),
        (axes[1], "Conditional CDF  F(x | Y = g)", f"{feature}: Conditional CDF"),
    ]:
        ax.set_xticks(x)
        ax.set_xticklabels(cat_labels, rotation=25, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    axes[1].legend(title="Obesity Level", frameon=True, fontsize=8)
    fig.tight_layout()

    path = os.path.join(out_dir, f"cond_cat_{_safe_name(feature)}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Continuous features: Conditional PDF (KDE) + Conditional CDF (ECDF)
# ---------------------------------------------------------------------------
def conditional_continuous_pdf_cdf(
    df: pd.DataFrame,
    feature: str,
    target_col: str,
    level_order: Optional[Sequence] = None,
    n_grid: int = 512,
    pad: float = 0.10,
) -> Dict[str, Dict[str, object]]:
    """
    Estimate the conditional PDF (Gaussian KDE, Scott's bandwidth) and
    empirical CDF of a continuous feature for each obesity level on a shared
    evaluation grid.

    Returns {obesity_level: dict} where each inner dict contains:
      grid : shared evaluation grid (global observed range extended by `pad`)
      pdf  : conditional KDE density evaluated on `grid`
      cdf  : conditional empirical CDF F_{n_g} evaluated on `grid`
      kde  : fitted scipy.stats.gaussian_kde object
      x    : sorted sample for that obesity level
      ecdf : ECDF at the sorted sample points (i / n_g)
      n    : stratum sample size n_g
    """
    sub = df[[feature, target_col]].dropna()
    levels = _resolve_levels(sub[target_col], level_order)

    all_x = sub[feature].to_numpy(dtype=float)
    lo, hi = float(np.min(all_x)), float(np.max(all_x))
    rng = hi - lo if hi > lo else 1.0
    grid = np.linspace(lo - pad * rng, hi + pad * rng, n_grid)

    results: Dict[str, Dict[str, object]] = {}
    for lvl in levels:
        x = np.sort(sub.loc[sub[target_col] == lvl, feature].to_numpy(dtype=float))
        n = len(x)
        kde = stats.gaussian_kde(x)  # Scott's rule by default
        pdf = kde(grid)
        cdf = np.searchsorted(x, grid, side="right") / n
        ecdf = np.arange(1, n + 1) / n
        results[lvl] = {
            "grid": grid,
            "pdf": pdf,
            "cdf": cdf,
            "kde": kde,
            "x": x,
            "ecdf": ecdf,
            "n": n,
        }

    return results


def conditional_support_intervals(
    df: pd.DataFrame,
    continuous_features: List[str],
    target_col: str,
    level_order: Optional[Sequence] = None,
) -> pd.DataFrame:
    """
    Compute observed bounds and 95% central quantile intervals [q2.5, q97.5]
    for every continuous feature, stratified by obesity level.
    """
    levels = _resolve_levels(df[target_col], level_order)
    rows = []
    for col in continuous_features:
        sub = df[[col, target_col]].dropna()
        for lvl in levels:
            x = sub.loc[sub[target_col] == lvl, col].to_numpy(dtype=float)
            q025, q25, med, q75, q975 = np.quantile(x, [0.025, 0.25, 0.50, 0.75, 0.975])
            rows.append(
                {
                    "feature": col,
                    "obesity_level": lvl,
                    "n": len(x),
                    "min": float(np.min(x)),
                    "q025": float(q025),
                    "q25": float(q25),
                    "median": float(med),
                    "q75": float(q75),
                    "q975": float(q975),
                    "max": float(np.max(x)),
                }
            )
    return pd.DataFrame(rows).set_index(["feature", "obesity_level"])


def plot_conditional_continuous(
    cond_dists: Dict[str, Dict[str, object]],
    feature: str,
    out_dir: str,
    level_order: Sequence,
) -> str:
    """Plot overlaid conditional KDE PDFs and conditional ECDFs across levels."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for lvl in level_order:
        if lvl not in cond_dists:
            continue
        d = cond_dists[lvl]
        color = LEVEL_COLORS.get(lvl, None)
        label = f"{str(lvl).replace('_', ' ')} (n={d['n']})"

        # Left panel: Conditional PDF (KDE)
        axes[0].plot(d["grid"], d["pdf"], label=label, color=color, linewidth=1.8)
        axes[0].fill_between(d["grid"], d["pdf"], alpha=0.12, color=color)

        # Right panel: Conditional CDF (ECDF)
        axes[1].step(
            d["x"], d["ecdf"], where="post", label=label, color=color, linewidth=1.8
        )

    axes[0].set_xlabel(feature)
    axes[0].set_ylabel("Conditional Density  f(x | Y = g)")
    axes[0].set_title(f"{feature}: Conditional PDF (Gaussian KDE)")
    axes[0].grid(linestyle="--", alpha=0.4)

    axes[1].set_xlabel(feature)
    axes[1].set_ylabel("Conditional CDF  F(x | Y = g)")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title(f"{feature}: Conditional ECDF")
    axes[1].grid(linestyle="--", alpha=0.4)
    axes[1].legend(title="Obesity Level", frameon=True, fontsize=8)

    fig.tight_layout()
    path = os.path.join(out_dir, f"cond_cont_{_safe_name(feature)}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------
def conditional_distributions(
    df: pd.DataFrame,
    target_col: str,
    categorical_features: List[str],
    continuous_features: List[str],
    out_dir: str = OUTPUT_DIR,
    level_order: Optional[Sequence] = DEFAULT_LEVEL_ORDER,
    category_orders: Optional[Dict[str, Sequence]] = None,
) -> Dict[str, object]:
    """
    Run Task 4 (Conditional Distributions) for every feature.

    Parameters
    ----------
    df                   : dataset (one column per feature + target).
    target_col           : name of the grouped 4-level obesity target column.
    categorical_features : names of categorical AND binary columns.
    continuous_features  : names of continuous columns.
    out_dir              : directory for figures and summary CSVs
                           (default figures/conditional-distributions).
    level_order          : ordering of the four obesity levels.
    category_orders      : optional {column: [ordered categories]} so that the
                           conditional CDF of ordinal variables (CAEC, CALC)
                           follows their natural order.

    Returns
    -------
    dict with
      'pmf_cdf_tables'          : {feature: DataFrame(count, n_g, pmf, cdf)}
      'continuous_distributions': {feature: {level: {grid, pdf, cdf, kde, x, ecdf, n}}}
      'support_intervals'       : DataFrame of observed bounds and 95% intervals
      'figures'                 : {feature: path to saved png}
    """
    category_orders = category_orders or {}
    required = [target_col, *categorical_features, *continuous_features]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Columns not found in dataframe: {missing}")

    os.makedirs(out_dir, exist_ok=True)
    tables_dir = os.path.join(out_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)

    levels = _resolve_levels(df[target_col], level_order)
    pmf_cdf_tables: Dict[str, pd.DataFrame] = {}
    continuous_dists: Dict[str, Dict[str, Dict[str, object]]] = {}
    figures: Dict[str, str] = {}

    for col in categorical_features:
        cat_order = _resolve_categories(df[col], category_orders.get(col))
        tbl = conditional_categorical_pmf_cdf(
            df, col, target_col, level_order=levels, category_order=cat_order
        )
        tbl.to_csv(os.path.join(tables_dir, f"cond_pmf_cdf_{_safe_name(col)}.csv"))
        pmf_cdf_tables[col] = tbl
        figures[col] = plot_conditional_categorical(
            tbl, col, out_dir, level_order=levels, category_order=cat_order
        )

    for col in continuous_features:
        dists = conditional_continuous_pdf_cdf(
            df, col, target_col, level_order=levels
        )
        continuous_dists[col] = dists
        figures[col] = plot_conditional_continuous(
            dists, col, out_dir, level_order=levels
        )

    support_intervals = conditional_support_intervals(
        df, continuous_features, target_col, level_order=levels
    )
    support_intervals.to_csv(os.path.join(out_dir, "conditional_support_intervals.csv"))

    return {
        "pmf_cdf_tables": pmf_cdf_tables,
        "continuous_distributions": continuous_dists,
        "support_intervals": support_intervals,
        "figures": figures,
    }


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    DATA_PATH = "ObesityDataSet_raw_and_data_sinthetic.csv"
    df = pd.read_csv(DATA_PATH)

    # Recode 7 original obesity classes into the 4 grouped levels
    level_map = {
        "Insufficient_Weight": "Insufficient_Weight",
        "Normal_Weight": "Normal_Weight",
        "Overweight_Level_I": "Overweight",
        "Overweight_Level_II": "Overweight",
        "Obesity_Type_I": "Obesity",
        "Obesity_Type_II": "Obesity",
        "Obesity_Type_III": "Obesity",
    }
    df["Obesity_Group"] = df["NObeyesdad"].map(level_map)

    categorical_cols = [
        "Gender",
        "family_history_with_overweight",
        "FAVC",
        "CAEC",
        "SMOKE",
        "SCC",
        "CALC",
        "MTRANS",
    ]
    continuous_cols = [
        "Age",
        "Height",
        "Weight",
        "FCVC",
        "NCP",
        "CH2O",
        "FAF",
        "TUE",
    ]
    ordinal_orders = {
        "CAEC": ["no", "Sometimes", "Frequently", "Always"],
        "CALC": ["no", "Sometimes", "Frequently", "Always"],
    }

    results = conditional_distributions(
        df=df,
        target_col="Obesity_Group",
        categorical_features=categorical_cols,
        continuous_features=continuous_cols,
        category_orders=ordinal_orders,
    )

    print("Saved conditional figures:", len(results["figures"]))
    print("\nConditional support intervals (first 8 rows):")
    print(results["support_intervals"].head(8).round(3))