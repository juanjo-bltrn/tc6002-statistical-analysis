import itertools
import os
import re
from typing import Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")  # Figures are saved, not shown
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

OUTPUT_DIR = os.path.join("figures", "categorical-features")

# ---------------------------------------------------------------------------
# Target levels (the target is ALREADY grouped into four categories)
# ---------------------------------------------------------------------------
# Edit these labels if your grouped column uses different spellings.
LEVELS = ["Insufficient_Weight", "Normal_Weight", "Overweight", "Obesity"]

# Colorblind-checked categorical slots 1-4 (blue, orange, aqua, yellow), fixed order
COLORS = dict(zip(LEVELS, ["#257de0", "#f47230", "#00b894", "#f2a600"]))
INK, INK_SOFT, GRID = "#191919", "#666666", "#e6e6e6"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def prepare_target(
    df: pd.DataFrame, target: str = "NObeyesdad", new_col: str = "obesity_level"
) -> pd.DataFrame:
    """
    Validate that the target is already grouped into the four levels in LEVELS
    and store them as an ordered categorical column `new_col`.
    """
    out = df.copy()
    found = set(out[target].dropna().unique())
    if found != set(LEVELS):
        raise ValueError(
            f"Column '{target}' must contain exactly the labels {LEVELS}; "
            f"found {sorted(map(str, found))}. Edit LEVELS at the top of "
            "this file if your labels are spelled differently."
        )
    out[new_col] = pd.Categorical(out[target], categories=LEVELS, ordered=True)
    return out


def _fmt_p(p: float) -> str:
    return "< 0.001" if p < 1e-3 else f"= {p:.3f}"


def _safe(name: str) -> str:
    """File-system friendly version of a column name."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name))


def _style(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(INK_SOFT)
    ax.spines["bottom"].set_color(INK_SOFT)
    ax.tick_params(colors=INK_SOFT, labelsize=9)
    ax.grid(axis="y", color=GRID, linestyle="--", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)


def bh_adjust(pvals: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg FDR adjustment for a sequence of p-values."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0.0, 1.0)
    out = np.empty(n, dtype=float)
    out[order] = adj
    return out


def _wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score 95% confidence interval for a binomial proportion."""
    if n == 0:
        return np.nan, np.nan
    z = stats.norm.ppf(1.0 - alpha / 2.0)
    p_hat = k / n
    denom = 1.0 + z**2 / n
    center = (p_hat + z**2 / (2.0 * n)) / denom
    half = (z * np.sqrt((p_hat * (1.0 - p_hat) + z**2 / (4.0 * n)) / n)) / denom
    lower = max(0.0, min(p_hat, center - half))
    upper = min(1.0, max(p_hat, center + half))
    return lower, upper


def _contingency_table(
    df: pd.DataFrame,
    feature: str,
    level_col: str = "obesity_level",
    order: Optional[Sequence] = None,
) -> pd.DataFrame:
    """Return a (levels x categories) contingency table of counts."""
    sub = df[[level_col, feature]].dropna()
    ct = pd.crosstab(sub[level_col], sub[feature]).reindex(index=LEVELS, fill_value=0)
    if order is not None:
        cols = [c for c in order if c in ct.columns]
        extra = [c for c in ct.columns if c not in cols]
        ct = ct[cols + extra]
    return ct


# ---------------------------------------------------------------------------
# (a) Conditional prevalence (proportions) with 95% Wilson CIs
# ---------------------------------------------------------------------------
def conditional_prevalence(
    df: pd.DataFrame,
    feature: str,
    level_col: str = "obesity_level",
    order: Optional[Sequence] = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Estimate conditional prevalence P(X = c | Y = level) with 95% Wilson CIs
    for each category `c` of `feature`, stratified by the four obesity levels.
    """
    ct = _contingency_table(df, feature, level_col, order)
    rows = []
    for lv in ct.index:
        n_lv = int(ct.loc[lv].sum())
        for cat in ct.columns:
            k = int(ct.loc[lv, cat])
            prop = k / n_lv if n_lv > 0 else np.nan
            lo, hi = _wilson_ci(k, n_lv, alpha=alpha)
            rows.append({
                "feature": feature,
                "level": lv,
                "category": cat,
                "count": k,
                "n_level": n_lv,
                "prevalence": prop,
                "ci_low": lo,
                "ci_high": hi,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# (b) Chi-squared, G-test (ANOFA equivalent), and Monte Carlo exact test
# ---------------------------------------------------------------------------
def _monte_carlo_chi2_p(
    obs: np.ndarray, n_sim: int = 5000, rng: Optional[np.random.Generator] = None
) -> float:
    """
    Monte Carlo permutation p-value for a contingency table with fixed
    row and column margins (used when expected cell counts < 5).
    """
    rng = rng or np.random.default_rng(42)
    obs_stat = stats.chi2_contingency(obs, correction=False).statistic
    row_sums = obs.sum(axis=1).astype(int)
    col_sums = obs.sum(axis=0).astype(int)
    n_total = int(row_sums.sum())

    # Expand column labels and permute across rows
    col_pool = np.repeat(np.arange(len(col_sums)), col_sums)
    split_idx = np.cumsum(row_sums)[:-1]
    n_cols = len(col_sums)

    exceed = 0
    for _ in range(n_sim):
        perm = rng.permutation(col_pool)
        chunks = np.split(perm, split_idx)
        sim_table = np.vstack([np.bincount(ch, minlength=n_cols) for ch in chunks])
        sim_stat = stats.chi2_contingency(sim_table, correction=False).statistic
        if sim_stat >= obs_stat - 1e-12:
            exceed += 1
    return (exceed + 1) / (n_sim + 1)


def test_categorical_feature(
    df: pd.DataFrame,
    feature: str,
    level_col: str = "obesity_level",
    order: Optional[Sequence] = None,
    n_sim: int = 5000,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, object]:
    """
    Perform Pearson's chi-squared test and log-likelihood ratio G-test across
    the four obesity levels. If any expected cell count < 5, also compute a
    Monte Carlo exact p-value and flag it as the recommended test.
    """
    ct = _contingency_table(df, feature, level_col, order)
    obs = ct.to_numpy(dtype=float)
    n = obs.sum()

    chi2_res = stats.chi2_contingency(obs, correction=False)
    g_res = stats.chi2_contingency(obs, lambda_="log-likelihood", correction=False)

    min_exp = float(chi2_res.expected_freq.min())
    low_exp_cells = int((chi2_res.expected_freq < 5.0).sum())
    pct_low_exp = 100.0 * low_exp_cells / obs.size

    # Cramér's V effect size
    r, c = obs.shape
    cramers_v = float(np.sqrt(chi2_res.statistic / (n * min(r - 1, c - 1))))

    # Standardized Pearson residuals: (O - E) / sqrt(E)
    std_resid = pd.DataFrame(
        (obs - chi2_res.expected_freq) / np.sqrt(chi2_res.expected_freq),
        index=ct.index,
        columns=ct.columns,
    )

    if low_exp_cells > 0:
        mc_p = _monte_carlo_chi2_p(obs, n_sim=n_sim, rng=rng)
        primary_test = "Monte Carlo Exact Chi2"
        primary_p = mc_p
    else:
        mc_p = np.nan
        primary_test = "Pearson Chi2"
        primary_p = float(chi2_res.pvalue)

    row = {
        "feature": feature,
        "type": "binary" if c == 2 else "categorical",
        "n_categories": c,
        "n_obs": int(n),
        "df": int(chi2_res.dof),
        "chi2_stat": float(chi2_res.statistic),
        "chi2_p": float(chi2_res.pvalue),
        "g_stat": float(g_res.statistic),
        "g_p": float(g_res.pvalue),
        "cramers_v": cramers_v,
        "min_expected": min_exp,
        "pct_expected_lt5": pct_low_exp,
        "mc_exact_p": mc_p,
        "primary_test": primary_test,
        "primary_p": primary_p,
    }
    return {"row": row, "contingency": ct, "std_resid": std_resid}


# ---------------------------------------------------------------------------
# (c) Post-hoc pairwise comparisons (chi-squared / Fisher's exact)
# ---------------------------------------------------------------------------
def posthoc_categorical(
    df: pd.DataFrame,
    feature: str,
    level_col: str = "obesity_level",
    order: Optional[Sequence] = None,
    n_sim: int = 2000,
    rng: Optional[np.random.Generator] = None,
) -> pd.DataFrame:
    """
    Pairwise comparisons across the 6 pairs of obesity levels.
    Uses Fisher's exact test for 2x2 tables (binary features) when any expected
    count < 5, Monte Carlo exact chi2 for 2xC tables with low counts, and
    Pearson's chi2 otherwise.
    """
    ct = _contingency_table(df, feature, level_col, order)
    rows = []
    for lv_i, lv_j in itertools.combinations(LEVELS, 2):
        sub = ct.loc[[lv_i, lv_j]]
        # Drop categories with 0 total count in this pair
        sub = sub.loc[:, sub.sum(axis=0) > 0]
        obs = sub.to_numpy(dtype=float)

        if obs.shape[1] < 2:
            rows.append({
                "feature": feature, "level_i": lv_i, "level_j": lv_j,
                "test": "constant", "stat": 0.0, "df": 0, "p_value": 1.0,
            })
            continue

        chi2_res = stats.chi2_contingency(obs, correction=False)
        low_exp = (chi2_res.expected_freq < 5.0).any()

        if obs.shape == (2, 2) and low_exp:
            odds, p_val = stats.fisher_exact(obs)
            test_name, stat_val, dof = "Fisher exact", float(odds), 1
        elif low_exp:
            p_val = _monte_carlo_chi2_p(obs, n_sim=n_sim, rng=rng)
            test_name = "Monte Carlo Chi2"
            stat_val, dof = float(chi2_res.statistic), int(chi2_res.dof)
        else:
            p_val = float(chi2_res.pvalue)
            test_name = "Pearson Chi2"
            stat_val, dof = float(chi2_res.statistic), int(chi2_res.dof)

        rows.append({
            "feature": feature,
            "level_i": lv_i,
            "level_j": lv_j,
            "test": test_name,
            "stat": stat_val,
            "df": dof,
            "p_value": p_val,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# (d) One-vs-rest diagnostic metrics & Cohen's kappa (Autoevaluation checklist)
# ---------------------------------------------------------------------------
def diagnostic_metrics_categorical(
    df: pd.DataFrame,
    feature: str,
    level_col: str = "obesity_level",
    order: Optional[Sequence] = None,
) -> pd.DataFrame:
    """
    For each (category, obesity_level) pair, treat `feature == category` as a
    binary indicator for `obesity_level` (one-vs-rest) and compute accuracy,
    sensitivity, specificity, and Cohen's kappa with an asymptotic z-test p-value.
    """
    sub = df[[level_col, feature]].dropna()
    cats = list(order) if order is not None else sorted(sub[feature].unique(), key=str)
    n = len(sub)
    rows = []

    for cat in cats:
        pred = (sub[feature] == cat).to_numpy()
        for lv in LEVELS:
            actual = (sub[level_col] == lv).to_numpy()
            tp = int(np.sum(pred & actual))
            fp = int(np.sum(pred & ~actual))
            fn = int(np.sum(~pred & actual))
            tn = int(np.sum(~pred & ~actual))

            acc = (tp + tn) / n
            sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
            spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan

            # Cohen's kappa and asymptotic standard error under H0: kappa = 0
            p_o = acc
            p_e = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (n**2)
            if p_e < 1.0:
                kappa = (p_o - p_e) / (1.0 - p_e)
                se0 = np.sqrt(p_e / (n * (1.0 - p_e))) if p_e > 0 else np.nan
                p_kappa = (
                    float(2.0 * stats.norm.sf(abs(kappa / se0)))
                    if (se0 is not None and not np.isnan(se0) and se0 > 0)
                    else np.nan
                )
            else:
                kappa, p_kappa = np.nan, np.nan

            rows.append({
                "feature": feature,
                "category": cat,
                "target_level": lv,
                "accuracy": acc,
                "sensitivity": sens,
                "specificity": spec,
                "cohens_kappa": kappa,
                "kappa_p": p_kappa,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def plot_categorical_feature(
    prev_df: pd.DataFrame,
    std_resid: pd.DataFrame,
    test_row: Dict[str, object],
    out_dir: str = OUTPUT_DIR,
) -> str:
    """
    Two-panel figure per feature:
      (1) Grouped bar chart of conditional prevalences with 95% Wilson CIs
      (2) Heatmap of standardized Pearson residuals (highlighting |z| > 2)
    """
    feature = str(test_row["feature"])
    cats = list(std_resid.columns)
    short_lv = [lv.replace("_", "\n") for lv in LEVELS]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), gridspec_kw={"width_ratios": [1.35, 1.0]})

    # (1) Grouped bars of conditional prevalence
    ax = axes[0]
    x = np.arange(len(cats))
    width = 0.8 / len(LEVELS)
    for i, lv in enumerate(LEVELS):
        sub = prev_df[prev_df["level"] == lv].set_index("category").reindex(cats)
        vals = sub["prevalence"].to_numpy()
        yerr_lo = vals - sub["ci_low"].to_numpy()
        yerr_hi = sub["ci_high"].to_numpy() - vals
        offset = (i - (len(LEVELS) - 1) / 2.0) * width
        ax.bar(
            x + offset, vals, width=width * 0.92, color=COLORS[lv],
            edgecolor=INK, linewidth=0.6, alpha=0.88, label=lv.replace("_", " "),
        )
        ax.errorbar(
            x + offset, vals, yerr=[yerr_lo, yerr_hi],
            fmt="none", ecolor=INK, elinewidth=1.0, capsize=2.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([str(c).replace("_", "\n") for c in cats], fontsize=8.5)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Conditional prevalence P(X = c | Y)", color=INK)
    ax.set_title("Conditional prevalence with 95% Wilson CI", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper right")
    _style(ax)

    # (2) Standardized residuals heatmap
    ax = axes[1]
    mat = std_resid.to_numpy(dtype=float)
    vmax = max(3.0, float(np.nanmax(np.abs(mat))))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(cats)))
    ax.set_xticklabels([str(c).replace("_", "\n") for c in cats], fontsize=8.5)
    ax.set_yticks(np.arange(len(LEVELS)))
    ax.set_yticklabels(short_lv, fontsize=8.5)
    ax.set_title("Standardized Pearson residuals", fontsize=10, color=INK)

    for r in range(mat.shape[0]):
        for c in range(mat.shape[1]):
            val = mat[r, c]
            txt_color = "white" if abs(val) > 0.55 * vmax else INK
            weight = "bold" if abs(val) >= 2.0 else "normal"
            ax.text(c, r, f"{val:+.2f}", ha="center", va="center",
                    color=txt_color, fontsize=8.5, fontweight=weight)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8, colors=INK_SOFT)

    fig.suptitle(
        f"{feature}: χ²({int(test_row['df'])}) = {test_row['chi2_stat']:.2f}, "
        f"p {_fmt_p(float(test_row['primary_p']))} ({test_row['primary_test']}), "
        f"Cramér's V = {test_row['cramers_v']:.3f}",
        y=1.03, color=INK,
    )
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{_safe(feature)}_by_obesity_level.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_effect_sizes(test_df: pd.DataFrame, out_dir: str = OUTPUT_DIR) -> str:
    """Horizontal bar chart of Cramér's V across all binary/categorical features."""
    df_s = test_df.sort_values("cramers_v", ascending=True)
    fig, ax = plt.subplots(figsize=(7.5, max(3.2, 0.48 * len(df_s))))
    colors = ["#257de0" if sig else "#b2bec3" for sig in df_s["significant_fdr"]]

    ax.barh(df_s["feature"], df_s["cramers_v"], color=colors, edgecolor=INK, linewidth=0.6)
    for i, (_, r) in enumerate(df_s.iterrows()):
        ax.text(
            r["cramers_v"] + 0.008, i,
            f"V = {r['cramers_v']:.3f}  (p_FDR {_fmt_p(r['p_fdr'])})",
            va="center", fontsize=8.5, color=INK,
        )

    ax.set_xlim(0, max(0.35, float(df_s["cramers_v"].max()) * 1.38))
    ax.set_xlabel("Cramér's V (association strength with obesity level)", color=INK)
    ax.set_title("Effect sizes of binary & categorical features", fontsize=10.5, color=INK)
    _style(ax)
    ax.grid(axis="x", color=GRID, linestyle="--", linewidth=0.7, alpha=0.8)
    ax.grid(axis="y", visible=False)

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "categorical_effect_sizes.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def analyze_categorical_features(
    df: pd.DataFrame,
    categorical_features: List[str],
    target: str = "NObeyesdad",
    out_dir: str = OUTPUT_DIR,
    alpha: float = 0.05,
    category_orders: Optional[Dict[str, Sequence]] = None,
    n_sim: int = 5000,
    seed: int = 42,
) -> Dict[str, object]:
    """
    Run Task 3 for all binary and categorical features.

    Returns dict with DataFrames:
      - 'prevalence'   : conditional prevalences & 95% Wilson CIs
      - 'tests'        : chi-squared, G-test, Cramér's V, exact p-values, BH FDR
      - 'posthoc'      : pairwise level comparisons (BH FDR) for significant features
      - 'diagnostics'  : one-vs-rest accuracy, sensitivity, specificity, Cohen's kappa
      - 'figures'      : list of saved PNG paths
    CSV copies are written to `out_dir`.
    """
    os.makedirs(out_dir, exist_ok=True)
    data = prepare_target(df, target)
    category_orders = category_orders or {}
    rng = np.random.default_rng(seed)

    prev_list, test_rows, diag_list = [], [], []
    resids_map, prev_map = {}, {}

    for f in categorical_features:
        order = category_orders.get(f)
        pv = conditional_prevalence(data, f, order=order, alpha=alpha)
        tr = test_categorical_feature(data, f, order=order, n_sim=n_sim, rng=rng)
        dg = diagnostic_metrics_categorical(data, f, order=order)

        prev_list.append(pv)
        test_rows.append(tr["row"])
        diag_list.append(dg)
        prev_map[f] = pv
        resids_map[f] = tr["std_resid"]

    prev_df = pd.concat(prev_list, ignore_index=True)
    test_df = pd.DataFrame(test_rows)
    test_df["p_fdr"] = bh_adjust(test_df["primary_p"])
    test_df["chi2_p_fdr"] = bh_adjust(test_df["chi2_p"])
    test_df["g_p_fdr"] = bh_adjust(test_df["g_p"])
    test_df["significant_raw"] = test_df["primary_p"] < alpha
    test_df["significant_fdr"] = test_df["p_fdr"] < alpha

    diag_df = pd.concat(diag_list, ignore_index=True)
    diag_df["kappa_p_fdr"] = bh_adjust(diag_df["kappa_p"].fillna(1.0))

    # Post-hoc pairwise tests only for features significant after FDR
    sig_features = test_df.loc[test_df["significant_fdr"], "feature"].tolist()
    ph_list = [
        posthoc_categorical(
            data, f, order=category_orders.get(f), n_sim=min(n_sim, 2000), rng=rng
        )
        for f in sig_features
    ]
    if ph_list:
        ph_df = pd.concat(ph_list, ignore_index=True)
        ph_df["p_fdr"] = bh_adjust(ph_df["p_value"])
        ph_df["significant_fdr"] = ph_df["p_fdr"] < alpha
    else:
        ph_df = pd.DataFrame()

    # Export CSVs
    prev_df.to_csv(os.path.join(out_dir, "conditional_prevalence.csv"), index=False)
    test_df.to_csv(os.path.join(out_dir, "chi2_g_test_results.csv"), index=False)
    ph_df.to_csv(os.path.join(out_dir, "posthoc_pairwise.csv"), index=False)
    diag_df.to_csv(os.path.join(out_dir, "diagnostic_metrics.csv"), index=False)

    # Generate figures
    test_idx = test_df.set_index("feature")
    figures = [
        plot_categorical_feature(
            prev_map[f],
            resids_map[f],
            {"feature": f, **test_idx.loc[f].to_dict()},
            out_dir=out_dir,
        )
        for f in categorical_features
    ]
    figures.append(plot_effect_sizes(test_df, out_dir))

    return {
        "prevalence": prev_df,
        "tests": test_df,
        "posthoc": ph_df,
        "diagnostics": diag_df,
        "figures": figures,
    }


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Load your dataset (with NObeyesdad already recoded into the 4 LEVELS)
    # df = pd.read_csv("obesity_grouped.csv")
    #
    # categorical_features = [
    #     "Gender", "family_history_with_overweight", "FAVC",
    #     "CAEC", "SMOKE", "SCC", "CALC", "MTRANS",
    # ]
    # category_orders = {
    #     "CAEC": ["no", "Sometimes", "Frequently", "Always"],
    #     "CALC": ["no", "Sometimes", "Frequently", "Always"],
    # }
    # results = analyze_categorical_features(
    #     df, categorical_features, category_orders=category_orders
    # )
    # print(results["tests"][["feature", "chi2_stat", "df", "primary_p", "p_fdr", "cramers_v"]])
    pass