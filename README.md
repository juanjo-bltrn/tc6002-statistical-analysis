# TC6002: Statistical Analysis of Obesity Levels Dataset

by Juan José H. Beltrán, Ph.D. in Computer Sciences.

Statistical analysis of the UCI *Estimation of Obesity Levels Based on Eating Habits and Physical Condition* dataset, which contains 2,111 records, 16 features (8 numerical, 8 categorical) and 1 target. The goal is to determine how the features relate to the obesity level and which ones are most useful to determine it statistically.

The target is recoded into four levels: Insufficient Weight (272), Normal Weight (287), Overweight (580, Levels I and II merged) and Obesity (972, Types I to III merged). All sections below are discussed in the accompanying report.

## Analysis Workflow

[`scripts/perform-analysis.ipynb`](scripts/perform-analysis.ipynb) runs the full analysis, using the modules in `scripts/`:

| Step | Module | Methods |
|------|--------|---------|
| Marginal distributions | `marginal_distributions.py` | PMF/CDF for categorical features; Gaussian KDE (Scott's rule) and ECDF for continuous ones; empirical supports. |
| Continuous features | `continuous_features_analysis.py` | Conditional means/variances with bootstrap 95% CIs; one-way ANOVA ($\eta^2$) with Welch and Kruskal–Wallis checks; Shapiro–Wilk, Brown–Forsythe, Q–Q plots; Tukey HSD and Welch post-hoc tests; Benjamini–Hochberg FDR. |
| Categorical features | `categorical_features_analysis.py` | Conditional prevalences with Wilson CIs; Pearson $\chi^2$ and G-test (Monte Carlo exact test for sparse cells); Cramér's $V$; standardized residuals; pairwise post-hoc tests; one-vs-rest accuracy, sensitivity, specificity and Cohen's $\kappa$. |
| Conditional distributions | `conditional_distributions.py` | Conditional PMF/CDF and KDE/ECDF per obesity level; quartiles and 95% central intervals. |
| Model-based expectations | `model_based_expectations.py` | Conditional mean and variance from the KDE by numerical integration, compared with sample moments and the analytical KDE variance. |
| Discriminative evaluation | `bootstrap_evaluation.py` | One-vs-rest KS tests; discriminative intervals; ROC/AUC; Youden-optimal thresholds; Type I/II error and power; bootstrap 95% CIs (B = 1,000). |

Plots and CSV tables for each step are saved in [`figures/`](figures/).

## Repository Layout

```text
tc6002-statistical-analysis/
├── data/
│   ├── obesity.csv              # Source dataset
│   └── dictionary.csv           # Variable names and descriptions
├── scripts/
│   ├── perform-analysis.ipynb
│   ├── marginal_distributions.py
│   ├── continuous_features_analysis.py
│   ├── categorical_features_analysis.py
│   ├── conditional_distributions.py
│   ├── model_based_expectations.py
│   └── bootstrap_evaluation.py
├── figures/
│   ├── marginal-distributions/
│   ├── continuous-features/
│   ├── categorical-features/
│   ├── conditional-distributions/
│   ├── model-based-expectations/
│   └── bootstrap-evaluation/
└── README.md
```

## Setup

Python 3.13 is currently used by the repository virtual environment.

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install pandas numpy scipy matplotlib seaborn phik ydata-profiling jupyter
```

Run the notebook from the repository root:

```powershell
jupyter notebook scripts/perform-analysis.ipynb
```
