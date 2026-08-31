"""Loads the Dehejia-Wahba (1999, 2002) re-analysis of LaLonde (1986):
the classic causal-inference benchmark for showing that naive comparison
of non-equivalent groups can get the sign of a causal effect wrong, and
that matching/reweighting can (partially) fix it.

- NSW treated / NSW experimental control: from a real randomized
  controlled trial (the National Supported Work Demonstration). The
  treated-vs-experimental-control difference is the RCT "gold standard"
  answer this project checks other methods against.
- CPS / PSID comparison groups: real (non-experimental) survey samples,
  NOT randomly assigned to treatment. Comparing treated vs. these
  directly is the "naive" analysis a real analyst might run before
  realizing the groups aren't comparable.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"

COVARIATES = ["age", "education", "black", "hispanic", "married", "nodegree", "re74", "re75"]
OUTCOME = "re78"


def load_nsw():
    df = pd.read_stata(DATA_DIR / "nsw_dw.dta")
    treated = df[df["treat"] == 1].reset_index(drop=True)
    experimental_control = df[df["treat"] == 0].reset_index(drop=True)
    return treated, experimental_control


def load_comparison_group(name):
    path = {"cps": "cps_controls.dta", "psid": "psid_controls.dta"}[name]
    df = pd.read_stata(DATA_DIR / path)
    return df.reset_index(drop=True)
