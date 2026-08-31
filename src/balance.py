"""Covariate balance diagnostics: standardized mean difference (SMD) per
covariate between treated and comparison group. This is the step a naive
causal-effect estimate skips -- and skipping it is exactly how you'd miss
that CPS/PSID are nothing like the treated group before ever looking at
outcomes.

Convention: SMD = (mean_treated - mean_comparison) / pooled_sd. |SMD| > 0.1
is a commonly used rule-of-thumb threshold for "meaningfully imbalanced."
"""
import numpy as np
import pandas as pd

from load_data import COVARIATES


def standardized_mean_diff(treated, comparison, covariates=COVARIATES, weights=None):
    rows = []
    for col in covariates:
        t_mean, t_var = treated[col].mean(), treated[col].var(ddof=1)
        if weights is None:
            c_mean, c_var = comparison[col].mean(), comparison[col].var(ddof=1)
        else:
            w = weights / weights.sum()
            c_mean = np.sum(w * comparison[col].values)
            c_var = np.sum(w * (comparison[col].values - c_mean) ** 2)
        pooled_sd = np.sqrt((t_var + c_var) / 2)
        smd = (t_mean - c_mean) / pooled_sd if pooled_sd > 0 else 0.0
        rows.append({"covariate": col, "treated_mean": t_mean, "comparison_mean": c_mean, "smd": smd})
    return pd.DataFrame(rows)
