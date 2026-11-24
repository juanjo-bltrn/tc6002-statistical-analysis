import os
import re
from typing import Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")  # Figures are saved, not shown

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

OUTPUT_DIR = os.path.join("figures", "marginal-distributions")


# ---------------------------------------------------------------------------
#                                 Helpers
# ---------------------------------------------------------------------------
def _safe_name(name: str) -> str:
    """File-system friendly version of a column name."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name))


def _save(fig: plt.Figure, out_dir: str, name: str, dpi: int = 150) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{_safe_name(name)}.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- 
#                       Discrete features: PMF + CDF
# ---------------------------------------------------------------------------
def categorical_pmf_cdf(
    series: pd.Series, order: Optional[Sequence] = None
) -> pd.DataFrame:
    """
    Empirical PMF and CDF of a discrete feature.

    Parameters
    ----------
    series : The observed values of the feature.
    order  : Optional ordering of the categories.

    Returns
    -------
    DataFrame indexed by category with columns: count, pmf, cdf.
    """
    s = series.dropna()
    counts = s.value_counts()
    if order is None:
        order = sorted(counts.index, key=lambda v: str(v))
    else:
        # keep any observed category that was not listed, at the end
        order = list(order) + [c for c in counts.index if c not in order]
    counts = counts.reindex(order, fill_value=0)

    out = pd.DataFrame({"count": counts})
    out["pmf"] = out["count"] / out["count"].sum()
    out["cdf"] = out["pmf"].cumsum()
    return out


def plot_categorical(
    series: pd.Series,
    name: str,
    out_dir: str = OUTPUT_DIR,
    order: Optional[Sequence] = None,
) -> str:
    """
    Bar plot of the PMF (left) and step plot of the CDF (right).
    """
    tab = categorical_pmf_cdf(series, order)
    labels = [str(i) for i in tab.index]
    x = np.arange(len(labels))

    fig, (ax_pmf, ax_cdf) = plt.subplots(1, 2, figsize=(11, 4))

    # PMF
    bars = ax_pmf.bar(x, tab["pmf"], color="#4C72B0", edgecolor="black")
    for b, p in zip(bars, tab["pmf"]):
        ax_pmf.text(b.get_x() + b.get_width() / 2, p, f"{p:.3f}",
                    ha="center", va="bottom", fontsize=8)
    ax_pmf.set_xticks(x, labels, rotation=30, ha="right")
    ax_pmf.set_ylim(0, tab["pmf"].max() * 1.15)
    ax_pmf.set_xlabel(name)
    ax_pmf.set_ylabel("P(X = x)")
    ax_pmf.set_title(f"PMF of {name}")

    # CDF (right-continuous step function over the ordered categories)
    ax_cdf.step(np.r_[-0.5, x], np.r_[0.0, tab["cdf"]], where="post",
                color="#C44E52", lw=2)
    ax_cdf.plot(x, tab["cdf"], "o", color="#C44E52")
    ax_cdf.set_xticks(x, labels, rotation=30, ha="right")
    ax_cdf.set_ylim(0, 1.05)
    ax_cdf.set_xlabel(name)
    ax_cdf.set_ylabel("P(X ≤ x)")
    ax_cdf.set_title(f"CDF of {name}")
    ax_cdf.grid(alpha=0.3)

    fig.suptitle(f"Marginal distribution of {name} (n = {int(tab['count'].sum())})",
                 y=1.02)
    return _save(fig, out_dir, f"{name}_pmf_cdf")


# ---------------------------------------------------------------------------
#               Continuous features: PDF (KDE) + CDF (ECDF)
# ---------------------------------------------------------------------------
def continuous_pdf_cdf(
    series: pd.Series, n_grid: int = 512, pad: float = 0.10
) -> Dict[str, object]:
    """
    Estimate the marginal PDF (Gaussian KDE, Scott's bandwidth) and the
    empirical CDF of a continuous feature.

    Returns a dict with
      grid : evaluation grid (observed range extended by `pad` on each side)
      pdf  : KDE density on `grid`
      cdf  : empirical CDF F_n evaluated on `grid`
      kde  : the fitted scipy gaussian_kde object
      x    : sorted sample
      ecdf : ECDF at the sorted sample points (i/n)
    """
    x = np.sort(series.dropna().to_numpy(dtype=float))
    n = len(x)
    lo, hi = x[0], x[-1]
    rng = hi - lo if hi > lo else 1.0
    grid = np.linspace(lo - pad * rng, hi + pad * rng, n_grid)

    kde = stats.gaussian_kde(x)
    pdf = kde(grid)
    cdf = np.searchsorted(x, grid, side="right") / n   # F_n(t) = #{x_i <= t}/n
    ecdf = np.arange(1, n + 1) / n
    return {"grid": grid, "pdf": pdf, "cdf": cdf, "kde": kde, "x": x, "ecdf": ecdf}


def plot_continuous(
    series: pd.Series, name: str, out_dir: str = OUTPUT_DIR, bins: str | int = "auto"
) -> str:
    """Histogram + KDE (left) and empirical CDF (right)."""
    est = continuous_pdf_cdf(series)
    x = est["x"]

    fig, (ax_pdf, ax_cdf) = plt.subplots(1, 2, figsize=(11, 4))

    # PDF
    ax_pdf.hist(x, bins=bins, density=True, color="#9ecae1", edgecolor="white",
                label="Histogram")
    ax_pdf.plot(est["grid"], est["pdf"], color="#4C72B0", lw=2, label="Gaussian KDE")
    ax_pdf.axvline(x.min(), color="gray", ls=":", lw=1)
    ax_pdf.axvline(x.max(), color="gray", ls=":", lw=1, label="Observed min / max")
    ax_pdf.set_xlabel(name)
    ax_pdf.set_ylabel("Density f(x)")
    ax_pdf.set_title(f"PDF of {name}")
    ax_pdf.legend(fontsize=8)

    # CDF
    ax_cdf.step(x, est["ecdf"], where="post", color="#C44E52", lw=2)
    ax_cdf.set_ylim(0, 1.05)
    ax_cdf.set_xlabel(name)
    ax_cdf.set_ylabel("P(X ≤ x)")
    ax_cdf.set_title(f"Empirical CDF of {name}")
    ax_cdf.grid(alpha=0.3)

    fig.suptitle(f"Marginal distribution of {name} (n = {len(x)})", y=1.02)
    return _save(fig, out_dir, f"{name}_pdf_cdf")


# ---------------------------------------------------------------------------
#                          Support description
# ---------------------------------------------------------------------------
def describe_support(
    df: pd.DataFrame,
    categorical_features: Sequence[str],
    continuous_features: Sequence[str],
    max_listed: int = 12,
) -> pd.DataFrame:
    """
    Table with the support (set of possible outcomes) of every feature.

    * categorical/binary : the finite set of observed categories.
    * continuous         : the observed interval [min, max]; the theoretical
                           support is bounded by the survey design / physical
                           limits and is stated only through the sample range.
    """
    rows = []
    for col in categorical_features:
        s = df[col].dropna()
        cats = sorted(s.unique(), key=lambda v: str(v))
        listed = ", ".join(map(str, cats[:max_listed]))
        if len(cats) > max_listed:
            listed += ", ..."
        kind = "binary" if len(cats) == 2 else "categorical"
        rows.append({
            "feature": col, "type": kind, "n_unique": len(cats),
            "support": "{" + listed + "}",
        })
    for col in continuous_features:
        s = df[col].dropna().astype(float)
        rows.append({
            "feature": col, "type": "continuous", "n_unique": s.nunique(),
            "support": f"[{s.min():.4g}, {s.max():.4g}]",
        })
    return pd.DataFrame(rows).set_index("feature")


# ---------------------------------------------------------------------------
#                                Main function
# ---------------------------------------------------------------------------
def marginal_distributions(
    df: pd.DataFrame,
    categorical_features: List[str],
    continuous_features: List[str],
    out_dir: str = OUTPUT_DIR,
    category_orders: Optional[Dict[str, Sequence]] = None,
) -> Dict[str, object]:
    """
    Run Task 1 for every feature.

    Parameters
    ----------
    df                   : dataset (one column per feature).
    categorical_features : names of categorical AND binary columns.
    continuous_features  : names of continuous columns.
    out_dir              : directory for the figures
                           (default figures/marginal-distributions).
    category_orders      : optional {column: [ordered categories]} so that the
                           CDF of ordinal variables (CAEC, CALC) is meaningful.

    Returns
    -------
    dict with
      'pmf_cdf_tables' : {feature: DataFrame(count, pmf, cdf)}   (categorical)
      'support'        : DataFrame describing the support of each feature
      'figures'        : {feature: path to saved png}
    """
    category_orders = category_orders or {}
    missing = [c for c in [*categorical_features, *continuous_features]
               if c not in df.columns]
    if missing:
        raise KeyError(f"Columns not found in dataframe: {missing}")

    os.makedirs(out_dir, exist_ok=True)
    tables, figures = {}, {}

    for col in categorical_features:
        order = category_orders.get(col)
        tables[col] = categorical_pmf_cdf(df[col], order)
        figures[col] = plot_categorical(df[col], col, out_dir, order)

    for col in continuous_features:
        figures[col] = plot_continuous(df[col], col, out_dir)

    support = describe_support(df, categorical_features, continuous_features)
    support.to_csv(os.path.join(out_dir, "support_summary.csv"))

    return {"pmf_cdf_tables": tables, "support": support, "figures": figures}