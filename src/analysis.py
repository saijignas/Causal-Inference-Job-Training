"""Main analysis: RCT gold-standard estimate, naive (non-experimental)
estimates, a diff-in-diff-style baseline-adjusted estimate, and matched/
IPW estimates -- all compared against the same RCT truth, with bootstrap
CIs so "close to the truth" is a statement with a margin of error, not
a vibe.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from balance import standardized_mean_diff
from load_data import COVARIATES, OUTCOME, load_comparison_group, load_nsw
from matching import fit_propensity, ipw_att, nn_match_att
from utils import bootstrap_ci, diff_in_means_ci, fmt_ci

ROOT = Path(__file__).parent.parent
FIG_DIR = ROOT / "results" / "figures"
TABLE_DIR = ROOT / "results" / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)


def naive_diff(treated, comparison, outcome=OUTCOME):
    return diff_in_means_ci(treated[outcome], comparison[outcome])


def did_style_diff(treated, comparison, outcome=OUTCOME, baseline="re75"):
    # Nets out pre-treatment earnings differences by using the *change*
    # in earnings (re78 - re75) as the outcome instead of the level --
    # a simple diff-in-diff, since re75 is measured before the program.
    t_change = treated[outcome] - treated[baseline]
    c_change = comparison[outcome] - comparison[baseline]
    return diff_in_means_ci(t_change, c_change)


def main():
    treated, experimental_control = load_nsw()
    print(f"NSW treated: {len(treated)}, NSW experimental control: {len(experimental_control)}")

    rct_ate, rct_ci = naive_diff(treated, experimental_control)
    print(f"\nRCT gold-standard ATE (treated vs. randomized control): {fmt_ci(rct_ate, rct_ci)}")

    rows = [{"method": "RCT (gold standard)", "comparison_group": "NSW experimental control",
             "estimate": rct_ate, "ci_half_width": rct_ci}]

    balance_frames = {"NSW experimental control": standardized_mean_diff(treated, experimental_control)}

    for group_name in ["cps", "psid"]:
        comparison = load_comparison_group(group_name)
        label = group_name.upper()
        print(f"\n=== {label} non-experimental comparison group (n={len(comparison)}) ===")

        balance_frames[label] = standardized_mean_diff(treated, comparison)
        max_smd = balance_frames[label]["smd"].abs().max()
        print(f"Max |SMD| before adjustment: {max_smd:.2f} (>0.1 is commonly flagged as imbalanced)")

        naive_ate, naive_ci = naive_diff(treated, comparison)
        print(f"Naive diff-in-means: {fmt_ci(naive_ate, naive_ci)}")
        rows.append({"method": "Naive diff-in-means", "comparison_group": label,
                     "estimate": naive_ate, "ci_half_width": naive_ci})

        did_ate, did_ci = did_style_diff(treated, comparison)
        print(f"Diff-in-diff style (re78-re75): {fmt_ci(did_ate, did_ci)}")
        rows.append({"method": "Diff-in-diff style", "comparison_group": label,
                      "estimate": did_ate, "ci_half_width": did_ci})

        match_att, matched_comparison, t_ps, c_ps = nn_match_att(treated, comparison)
        match_mean, match_lo, match_hi = bootstrap_ci(
            lambda t, c: nn_match_att(t, c)[0], treated, comparison
        )
        match_ci_half = (match_hi - match_lo) / 2
        print(f"Nearest-neighbor PS matching ATT: {fmt_ci(match_att, match_ci_half)}")
        rows.append({"method": "PS nearest-neighbor matching", "comparison_group": label,
                      "estimate": match_att, "ci_half_width": match_ci_half})

        balance_frames[f"{label} (post-matching)"] = standardized_mean_diff(
            treated, matched_comparison
        )

        ipw_estimate, weights, _, _ = ipw_att(treated, comparison)
        ipw_mean, ipw_lo, ipw_hi = bootstrap_ci(lambda t, c: ipw_att(t, c)[0], treated, comparison)
        ipw_ci_half = (ipw_hi - ipw_lo) / 2
        print(f"Inverse propensity weighting ATT: {fmt_ci(ipw_estimate, ipw_ci_half)}")
        rows.append({"method": "Inverse propensity weighting", "comparison_group": label,
                      "estimate": ipw_estimate, "ci_half_width": ipw_ci_half})

    results = pd.DataFrame(rows)
    results.to_csv(TABLE_DIR / "ate_estimates.csv", index=False)

    # ATE comparison plot
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {"cps": "#c44e52", "psid": "#dd8452", "NSW experimental control": "#4c72b0"}
    y_pos = np.arange(len(results))
    bar_colors = [
        colors.get(row["comparison_group"].lower().split()[0], colors["NSW experimental control"])
        for _, row in results.iterrows()
    ]
    ax.barh(y_pos, results["estimate"], xerr=results["ci_half_width"], color=bar_colors, capsize=3)
    ax.axvline(rct_ate, color="black", linestyle="--", linewidth=1, label=f"RCT truth (${rct_ate:,.0f})")
    labels = [f"{row['method']}\n({row['comparison_group']})" for _, row in results.iterrows()]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("estimated effect on 1978 earnings ($)")
    ax.set_title("Does the estimate get closer to the RCT truth as methods get more careful?")
    ax.legend()
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ate_comparison.png", dpi=150)
    print(f"\nSaved {FIG_DIR / 'ate_comparison.png'}")

    # Love plot: balance before vs after matching, for one comparison group (CPS)
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    before = balance_frames["CPS"].set_index("covariate")["smd"]
    after = balance_frames["CPS (post-matching)"].set_index("covariate")["smd"]
    y = np.arange(len(before))
    ax2.scatter(before, y, label="before matching", color="#c44e52", zorder=3)
    ax2.scatter(after, y, label="after matching", color="#4c72b0", zorder=3)
    for yi in y:
        ax2.plot([before.iloc[yi], after.iloc[yi]], [yi, yi], color="gray", linewidth=1, zorder=1)
    ax2.axvline(0, color="black", linewidth=0.8)
    ax2.axvline(0.1, color="gray", linestyle=":", linewidth=1)
    ax2.axvline(-0.1, color="gray", linestyle=":", linewidth=1)
    ax2.set_yticks(y)
    ax2.set_yticklabels(before.index)
    ax2.set_xlabel("standardized mean difference (treated vs. CPS)")
    ax2.set_title("Covariate balance before/after matching (dotted lines = ±0.1 rule of thumb)")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(FIG_DIR / "love_plot_cps.png", dpi=150)
    print(f"Saved {FIG_DIR / 'love_plot_cps.png'}")

    for name, frame in balance_frames.items():
        safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        frame.to_csv(TABLE_DIR / f"balance_{safe_name}.csv", index=False)


if __name__ == "__main__":
    main()
