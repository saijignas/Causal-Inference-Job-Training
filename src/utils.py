import numpy as np
from scipy import stats


def mean_ci(values, confidence=0.95):
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = values.mean()
    if n < 2:
        return mean, float("nan")
    sem = values.std(ddof=1) / np.sqrt(n)
    t_crit = stats.t.ppf((1 + confidence) / 2, df=n - 1)
    return mean, t_crit * sem


def fmt_ci(mean, half_width, decimals=0):
    return f"{mean:,.{decimals}f} ± {half_width:,.{decimals}f}"


def diff_in_means_ci(treated, control, confidence=0.95):
    """Welch's t-test style CI for a difference in means (no equal-variance
    assumption -- the treated/control groups here have very different
    sizes and variances)."""
    treated, control = np.asarray(treated, dtype=float), np.asarray(control, dtype=float)
    diff = treated.mean() - control.mean()
    se = np.sqrt(treated.var(ddof=1) / len(treated) + control.var(ddof=1) / len(control))
    df = len(treated) + len(control) - 2
    t_crit = stats.t.ppf((1 + confidence) / 2, df=df)
    return diff, t_crit * se


def bootstrap_ci(func, *arrays, n_boot=2000, confidence=0.95, random_state=42):
    """Percentile bootstrap CI for an estimator that isn't a simple
    closed-form statistic (e.g. a matching estimator)."""
    rng = np.random.RandomState(random_state)
    n = len(arrays[0])
    estimates = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        # pandas objects need .iloc for positional row selection --
        # arr[idx] on a DataFrame tries column lookup instead, silently
        # doing the wrong thing rather than erroring.
        estimates.append(func(*[a.iloc[idx] if hasattr(a, "iloc") else a[idx] for a in arrays]))
    estimates = np.array(estimates)
    alpha = (1 - confidence) / 2
    lo, hi = np.percentile(estimates, [100 * alpha, 100 * (1 - alpha)])
    return np.mean(estimates), lo, hi
