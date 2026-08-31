"""Two causal-estimation methods applied to the non-experimental
comparison groups (CPS, PSID), both trying to recover the RCT's answer
from data that wasn't randomized:

1. Propensity-score nearest-neighbor matching (ATT): for each treated
   unit, find the comparison unit with the closest propensity score,
   then take the mean outcome difference over matched pairs.
2. Inverse propensity weighting (ATT): reweight comparison units by
   p/(1-p) so the weighted comparison group's covariate distribution
   resembles the treated group's.

Both rely on the same assumption: conditional on the observed covariates
(crucially, including re74/re75 -- pre-treatment earnings), treatment
assignment is "as good as random." That assumption is checkable via
balance (balance.py), not provable -- disclosed as a limitation, not
glossed over.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

from load_data import COVARIATES, OUTCOME


def fit_propensity(treated, comparison, covariates=COVARIATES):
    combined = pd.concat([treated, comparison], ignore_index=True)
    X = combined[covariates].values
    y = np.r_[np.ones(len(treated)), np.zeros(len(comparison))]
    model = LogisticRegression(max_iter=2000)
    model.fit(X, y)
    treated_ps = model.predict_proba(treated[covariates].values)[:, 1]
    comparison_ps = model.predict_proba(comparison[covariates].values)[:, 1]
    return treated_ps, comparison_ps


def nn_match_att(treated, comparison, covariates=COVARIATES, outcome=OUTCOME):
    treated_ps, comparison_ps = fit_propensity(treated, comparison, covariates)
    nn = NearestNeighbors(n_neighbors=1).fit(comparison_ps.reshape(-1, 1))
    _, idx = nn.kneighbors(treated_ps.reshape(-1, 1))
    matched_comparison = comparison.iloc[idx.ravel()].reset_index(drop=True)
    att = treated[outcome].values - matched_comparison[outcome].values
    return att.mean(), matched_comparison, treated_ps, comparison_ps


def ipw_att(treated, comparison, covariates=COVARIATES, outcome=OUTCOME):
    treated_ps, comparison_ps = fit_propensity(treated, comparison, covariates)
    # ATT weights: treated get weight 1; comparison get p/(1-p), clipped to
    # avoid explosive weights from near-1 propensity scores (a known IPW
    # failure mode when treated/comparison overlap poorly).
    comparison_ps_clipped = np.clip(comparison_ps, 0.001, 0.999)
    weights = comparison_ps_clipped / (1 - comparison_ps_clipped)
    weighted_comparison_mean = np.sum(weights * comparison[outcome].values) / weights.sum()
    att = treated[outcome].mean() - weighted_comparison_mean
    return att, weights, treated_ps, comparison_ps
