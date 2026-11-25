import itertools
import os
import re
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

OUTPUT_DIR = os.path.join("figures", "continuous-features")

# --------------------------------------------------------------------------- #
# Target levels (the target is ALREADY grouped into four categories)
# --------------------------------------------------------------------------- #
# Edit these labels if your grouped column uses different spellings.
LEVELS = ['Insufficient_Weight', 'Normal_Weight', "Overweight", "Obesity"]
# Colorblind-checked categorical slots 1-4 (blue, orange, aqua, yellow), fixed order
COLORS = dict(zip(LEVELS, ["#257de0", "#f47230", "#00b894", "#f2a600"]))
INK, INK_SOFT, GRID = "#191919", "#666666", "#e6e6e6"


def prepare_target(df: pd.DataFrame, target: str = "NObeyesdad",
                   new_col: str = "obesity_level") -> pd.DataFrame:
    """
    The target is assumed to be already grouped into the four levels in LEVELS.
    This only validates the labels and stores them as an ordered categorical
    column `new_col` (no recoding is performed).
    """
    out = df.copy()
    found = set(out[target].dropna().unique())
    if found != set(LEVELS):
        raise ValueError(f"Column '{target}' must contain exactly the labels {LEVELS}; "
                         f"found {sorted(map(str, found))}. Edit LEVELS at the top of "
                         "this file if your labels are spelled differently.")
    out[new_col] = pd.Categorical(out[target], categories=LEVELS, ordered=True)
    return out


def _fmt_p(p: float) -> str:
    return "< 0.001" if p < 1e-3 else f"= {p:.3f}"


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name))


def _groups(df: pd.DataFrame, feature: str, level_col: str) -> Dict[str, np.ndarray]:
    return {lv: df.loc[df[level_col] == lv, feature].dropna().to_numpy(float)
            for lv in LEVELS}


# --------------------------------------------------------------------------- #
# (a) Conditional mean / variance with bootstrap CIs
# --------------------------------------------------------------------------- #
def conditional_stats(df: pd.DataFrame, feature: str, level_col: str = "obesity_level",
                      n_boot: int = 1000, ci: float = 0.95,
                      rng: Optional[np.random.Generator] = None) -> pd.DataFrame:
    """
    Per-level n, sample mean, unbiased sample variance (ddof=1), SD, empirical
    min/max, and percentile-bootstrap CIs for the mean and the variance.
    """
    rng = rng or np.random.default_rng(0)
    a = (1 - ci) / 2
    rows = []
    for lv, x in _groups(df, feature, level_col).items():
        n = len(x)
        idx = rng.integers(0, n, size=(n_boot, n))
        samp = x[idx]
        bm, bv = samp.mean(axis=1), samp.var(axis=1, ddof=1)
        rows.append({
            "feature": feature, "level": lv, "n": n,
            "mean": x.mean(), "mean_ci_low": np.quantile(bm, a),
            "mean_ci_high": np.quantile(bm, 1 - a),
            "variance": x.var(ddof=1), "var_ci_low": np.quantile(bv, a),
            "var_ci_high": np.quantile(bv, 1 - a),
            "sd": x.std(ddof=1), "min": x.min(), "max": x.max(),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# (b) One-way ANOVA (+ lm() cross-check, Welch, assumptions)
# --------------------------------------------------------------------------- #
def _classical_anova(groups: List[np.ndarray]) -> Dict[str, float]:
    k = len(groups)
    n_i = np.array([len(g) for g in groups])
    n = n_i.sum()
    grand = np.concatenate(groups).mean()
    means = np.array([g.mean() for g in groups])
    ssb = float(np.sum(n_i * (means - grand) ** 2))
    ssw = float(sum(((g - g.mean()) ** 2).sum() for g in groups))
    df1, df2 = k - 1, n - k
    F = (ssb / df1) / (ssw / df2)
    return dict(SSB=ssb, SSW=ssw, df_between=df1, df_within=df2, F=F,
                p=stats.f.sf(F, df1, df2), eta_sq=ssb / (ssb + ssw))


def _lm_anova(groups: List[np.ndarray]) -> Dict[str, float]:
    """F test of the overall regression  X ~ 1 + level  (lm() analogue)."""
    y = np.concatenate(groups)
    lab = np.concatenate([np.full(len(g), i) for i, g in enumerate(groups)])
    X = np.column_stack([np.ones_like(y)] + [(lab == j).astype(float)
                                              for j in range(1, len(groups))])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    rss, tss = float(resid @ resid), float(((y - y.mean()) ** 2).sum())
    p_par, n = X.shape[1], len(y)
    F = ((tss - rss) / (p_par - 1)) / (rss / (n - p_par))
    return dict(F=F, p=stats.f.sf(F, p_par - 1, n - p_par),
                r_squared=1 - rss / tss, residuals=resid)


def _welch_anova(groups: List[np.ndarray]) -> Dict[str, float]:
    k = len(groups)
    n_i = np.array([len(g) for g in groups], float)
    m_i = np.array([g.mean() for g in groups])
    v_i = np.array([g.var(ddof=1) for g in groups])
    w = n_i / v_i
    W = w.sum()
    m_w = (w * m_i).sum() / W
    lam = ((1 - w / W) ** 2 / (n_i - 1)).sum()
    F = ((w * (m_i - m_w) ** 2).sum() / (k - 1)) / (1 + 2 * (k - 2) / (k ** 2 - 1) * lam)
    df2 = (k ** 2 - 1) / (3 * lam)
    return dict(F=F, df1=k - 1, df2=df2, p=stats.f.sf(F, k - 1, df2))


def anova_feature(df: pd.DataFrame, feature: str, level_col: str = "obesity_level"
                  ) -> Dict[str, object]:
    g = list(_groups(df, feature, level_col).values())
    aov = _classical_anova(g)
    lm = _lm_anova(g)
    welch = _welch_anova(g)
    f_scipy = stats.f_oneway(*g)
    assert np.isclose(aov["F"], lm["F"]) and np.isclose(aov["F"], f_scipy.statistic), \
        "aov / lm / scipy F statistics disagree"

    resid = lm["residuals"]
    sw = stats.shapiro(resid)
    lev = stats.levene(*g, center="median")          # Brown-Forsythe
    kw = stats.kruskal(*g)
    row = {
        "feature": feature,
        "F": aov["F"], "df_between": aov["df_between"], "df_within": aov["df_within"],
        "p_value": aov["p"], "eta_sq": aov["eta_sq"],
        "F_lm": lm["F"], "R2_lm": lm["r_squared"],
        "welch_F": welch["F"], "welch_df1": welch["df1"], "welch_df2": welch["df2"],
        "welch_p": welch["p"],
        "kruskal_H": kw.statistic, "kruskal_p": kw.pvalue,
        "shapiro_W_resid": sw.statistic, "shapiro_p_resid": sw.pvalue,
        "levene_W": lev.statistic, "levene_p": lev.pvalue,
    }
    return {"row": row, "residuals": resid}


# --------------------------------------------------------------------------- #
# (c) Post-hoc pairwise comparisons
# --------------------------------------------------------------------------- #
def posthoc_feature(df: pd.DataFrame, feature: str, level_col: str = "obesity_level",
                    ci: float = 0.95) -> pd.DataFrame:
    """
    Tukey HSD (already adjusted for the 6 comparisons within the feature, with
    simultaneous CIs) and Welch t-tests (unadjusted, robust to heteroscedasticity).
    Mean difference is level_i - level_j.
    """
    grp = _groups(df, feature, level_col)
    names = list(grp)
    arrs = list(grp.values())
    tk = stats.tukey_hsd(*arrs)
    tci = tk.confidence_interval(confidence_level=ci)
    rows = []
    for i, j in itertools.combinations(range(len(names)), 2):
        t = stats.ttest_ind(arrs[i], arrs[j], equal_var=False)
        rows.append({
            "feature": feature, "level_i": names[i], "level_j": names[j],
            "mean_diff": arrs[i].mean() - arrs[j].mean(),
            "tukey_ci_low": tci.low[i, j], "tukey_ci_high": tci.high[i, j],
            "tukey_p_adj": tk.pvalue[i, j],
            "welch_t": t.statistic, "welch_df": t.df, "welch_p": t.pvalue,
        })
    return pd.DataFrame(rows)


def bh_adjust(p: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg FDR-adjusted p-values (q-values)."""
    p = np.asarray(p, float)
    return stats.false_discovery_control(p, method="bh") if len(p) else p


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_SOFT, labelsize=8)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def plot_feature(df: pd.DataFrame, feature: str, cstats: pd.DataFrame,
                 resid: np.ndarray, anova_row: Dict[str, float],
                 level_col: str = "obesity_level", out_dir: str = OUTPUT_DIR) -> str:
    """(1) boxplots by level, (2) mean +/- 95% bootstrap CI, (3) residual Q-Q plot."""
    grp = _groups(df, feature, level_col)
    short = [l.replace(" ", "\n") for l in LEVELS]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # (1) boxplot
    ax = axes[0]
    bp = ax.boxplot(list(grp.values()), tick_labels=short, patch_artist=True,
                    widths=0.55, showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="white",
                                   markeredgecolor=INK, markersize=5),
                    medianprops=dict(color=INK, lw=1.5),
                    flierprops=dict(marker="o", markersize=2, alpha=0.4,
                                    markeredgecolor=INK_SOFT))
    for patch, lv in zip(bp["boxes"], LEVELS):
        patch.set(facecolor=COLORS[lv], edgecolor=INK, alpha=0.85)
    ax.set_title("Distribution by obesity level", fontsize=10, color=INK)
    ax.set_ylabel(feature, color=INK)
    _style(ax)

    # (2) mean +- CI
    ax = axes[1]
    x = np.arange(len(LEVELS))
    for xi, (_, r) in zip(x, cstats.iterrows()):
        ax.errorbar(xi, r["mean"], yerr=[[r["mean"] - r["mean_ci_low"]],
                                         [r["mean_ci_high"] - r["mean"]]],
                    fmt="o", color=COLORS[r["level"]], ecolor=COLORS[r["level"]],
                    elinewidth=2, capsize=4, markersize=8,
                    markeredgecolor="white", markeredgewidth=1.5)
    ax.set_xticks(x, short)
    ax.set_xlim(-0.5, len(LEVELS) - 0.5)
    ax.set_title("Sample mean with 95% bootstrap CI", fontsize=10, color=INK)
    _style(ax)

    # (3) QQ of residuals
    ax = axes[2]
    (osm, osr), (slope, intercept, _) = stats.probplot(resid, dist="norm")
    ax.plot(osm, osr, ".", color="#257de0", markersize=3, alpha=0.6)
    ax.plot(osm, slope * osm + intercept, color=INK, lw=1.2)
    ax.set_xlabel("Theoretical quantiles", color=INK_SOFT)
    ax.set_ylabel("ANOVA residuals", color=INK_SOFT)
    ax.set_title(f"Residual Q-Q (Shapiro p = {anova_row['shapiro_p_resid']:.2g})",
                 fontsize=10, color=INK)
    _style(ax)

    fig.suptitle(
        f"{feature}: one-way ANOVA  F({int(anova_row['df_between'])}, "
        f"{int(anova_row['df_within'])}) = {anova_row['F']:.2f}, "
        f"p {_fmt_p(anova_row['p_value'])}, η² = {anova_row['eta_sq']:.3f}",
        y=1.03, color=INK)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{_safe(feature)}_by_obesity_level.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_effect_sizes(anova_df: pd.DataFrame, out_dir: str = OUTPUT_DIR) -> str:
    """Horizontal bars of eta^2 per feature, annotated with the FDR-adjusted p."""
    d = anova_df.sort_values("eta_sq")
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(d) + 1.5))
    bars = ax.barh(d["feature"], d["eta_sq"], color="#257de0", height=0.6)
    for b, q in zip(bars, d["p_fdr"]):
        ax.text(b.get_width(), b.get_y() + b.get_height() / 2,
                f"  q = {q:.2g}", va="center", fontsize=8, color=INK_SOFT)
    ax.set_xlim(0, d["eta_sq"].max() * 1.25)
    ax.set_xlabel("Effect size η² (share of variance explained by obesity level)",
                  color=INK)
    ax.set_title("ANOVA effect size by feature (q = BH-adjusted p)", color=INK)
    _style(ax)
    ax.yaxis.grid(False)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "anova_effect_sizes.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def analyze_continuous_features(
    df: pd.DataFrame,
    continuous_features: List[str],
    target: str = "NObeyesdad",
    out_dir: str = OUTPUT_DIR,
    alpha: float = 0.05,
    n_boot: int = 1000,
    seed: int = 42,
) -> Dict[str, object]:
    """
    Run Task 2 for all continuous features.

    Returns dict with DataFrames 'conditional_stats', 'anova', 'posthoc' and the
    list of saved figure paths. CSV copies are written to `out_dir`.

    FDR strategy
    ------------
    * ANOVA p-values: BH across the continuous features (column `p_fdr`).
    * Post-hoc: Tukey p-values are already adjusted within each feature
      (family-wise). BH is then applied to the pooled Tukey (and Welch) p-values
      of all comparisons made, controlling the FDR across features
      (columns `tukey_p_fdr`, `welch_p_fdr`).
    """
    os.makedirs(out_dir, exist_ok=True)
    data = prepare_target(df, target)
    rng = np.random.default_rng(seed)

    cond, an_rows, resids, figs_ctx = [], [], {}, {}
    for f in continuous_features:
        cs = conditional_stats(data, f, n_boot=n_boot, rng=rng)
        res = anova_feature(data, f)
        cond.append(cs)
        an_rows.append(res["row"])
        resids[f] = res["residuals"]
        figs_ctx[f] = cs

    cond_df = pd.concat(cond, ignore_index=True)
    an_df = pd.DataFrame(an_rows)
    an_df["p_fdr"] = bh_adjust(an_df["p_value"])
    an_df["significant_raw"] = an_df["p_value"] < alpha
    an_df["significant_fdr"] = an_df["p_fdr"] < alpha
    an_df["normal_resid_ok"] = an_df["shapiro_p_resid"] >= alpha
    an_df["equal_var_ok"] = an_df["levene_p"] >= alpha

    # post-hoc only where the ANOVA is significant (p < alpha), as required
    ph = [posthoc_feature(data, f) for f in an_df.loc[an_df["significant_raw"], "feature"]]
    if ph:
        ph_df = pd.concat(ph, ignore_index=True)
        ph_df["tukey_p_fdr"] = bh_adjust(ph_df["tukey_p_adj"])
        ph_df["welch_p_fdr"] = bh_adjust(ph_df["welch_p"])
        ph_df["tukey_sig_fdr"] = ph_df["tukey_p_fdr"] < alpha
        ph_df["welch_sig_fdr"] = ph_df["welch_p_fdr"] < alpha
    else:
        ph_df = pd.DataFrame()

    cond_df.to_csv(os.path.join(out_dir, "conditional_mean_variance.csv"), index=False)
    an_df.to_csv(os.path.join(out_dir, "anova_results.csv"), index=False)
    ph_df.to_csv(os.path.join(out_dir, "posthoc_pairwise.csv"), index=False)

    figures = [plot_feature(data, f, figs_ctx[f], resids[f],
                            an_df.set_index("feature").loc[f].to_dict(), out_dir=out_dir)
               for f in continuous_features]
    figures.append(plot_effect_sizes(an_df, out_dir))

    return {"conditional_stats": cond_df, "anova": an_df, "posthoc": ph_df,
            "figures": figures}