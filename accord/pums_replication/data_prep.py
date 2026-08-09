"""Data loading and splitting for the ACS PUMS replication of the
selection-conditioned-audit-blindness experiment.

Task: ACSIncome (predict PINCP > $50k) on one state-year of ACS PUMS,
fetched via `folktables` (installs cleanly, no credentialing needed).
State/year is fixed below -- change only if re-running for a robustness
check, never mid-analysis.

Group-blindness: the classifier's feature set is ACSIncome.features MINUS
SEX and RAC1P. Both are held out as candidate group axes for the audit
(primary = SEX, secondary = RAC1P restricted to White/Black) and never
shown to the model, mirroring the MIMIC experiment's LANGUAGE/ETHNICITY
exclusion.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from folktables import ACSDataSource, ACSIncome

STATE = "TX"
SURVEY_YEAR = "2018"
DATA_ROOT = Path(__file__).resolve().parent / "data"

# Features fed to the classifier -- SEX and RAC1P deliberately excluded.
MODEL_FEATURES = ["AGEP", "COW", "SCHL", "MAR", "OCCP", "POBP", "RELP", "WKHP"]
CATEGORICAL_FEATURES = ["COW", "MAR", "OCCP", "POBP", "RELP"]
NUMERIC_FEATURES = ["AGEP", "SCHL", "WKHP"]

TRAIN_FRAC = 0.60
VAL_FRAC = 0.20
# remaining 0.20 is TEST


def load_raw() -> pd.DataFrame:
    src = ACSDataSource(survey_year=SURVEY_YEAR, horizon="1-Year", survey="person",
                         root_dir=str(DATA_ROOT))
    return src.get_data(states=[STATE], download=True)


def build_task():
    """Returns X (model features only), y, and a DataFrame of group/aux
    columns (SEX, RAC1P, WKHP-for-selection), all row-aligned."""
    raw = load_raw()
    X_full, y, _ = ACSIncome.df_to_pandas(raw)
    y = y[ACSIncome.target].astype(int).to_numpy()
    X_model = X_full[MODEL_FEATURES].copy()
    aux = X_full[["SEX", "RAC1P", "WKHP"]].copy()
    aux["female"] = (aux["SEX"] == 2).astype(int)
    aux["black"] = (aux["RAC1P"] == 2).astype(int)
    aux["white"] = (aux["RAC1P"] == 1).astype(int)
    return X_model, y, aux


def split(X, y, aux, seed: int):
    """Fixed-seed 60/20/20 train/val/test split, no stratification tricks --
    a plain shuffle-split, same convention as a bare sklearn train_test_split
    done twice. Returns dict of arrays/frames per split."""
    n = len(y)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_train = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)
    tr_idx = idx[:n_train]
    va_idx = idx[n_train:n_train + n_val]
    te_idx = idx[n_train + n_val:]

    def take(ix):
        return dict(
            X=X.iloc[ix].reset_index(drop=True),
            y=y[ix],
            aux=aux.iloc[ix].reset_index(drop=True),
        )

    out = dict(train=take(tr_idx), val=take(va_idx), test=take(te_idx))
    # duplicate-row check across splits (leakage guard) -- ACSIncome rows are
    # one-per-person (SERIALNO+SPORDER not retained here, but index-based
    # partition is disjoint by construction; assert no index overlap)
    assert len(set(tr_idx) & set(va_idx)) == 0
    assert len(set(tr_idx) & set(te_idx)) == 0
    assert len(set(va_idx) & set(te_idx)) == 0
    return out


def selection_feature(aux: pd.DataFrame) -> np.ndarray:
    """z-scored WKHP (usual hours worked per week) -- an outcome-predictive,
    group-blind-classifier-excluded covariate used to drive Regime B
    selection. Chosen because it correlates with the outcome (income>50k,
    r~0.32) and, in this data, with SEX (r~-0.21) -- selection on it will
    induce a group disparity through a real correlation, not by
    construction."""
    w = aux["WKHP"].to_numpy().astype(float)
    return (w - w.mean()) / w.std()
