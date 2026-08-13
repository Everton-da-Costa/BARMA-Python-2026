"""Unit tests for the BARMA log-likelihood calculation.

This module validates the `loglik_barma` implementation against a reference
value computed in R, ensuring the Python port reproduces the expected log-
likelihood for a fixed dataset and set of parameters.

The tests rely on CSV fixtures in `data/raw/` that contain both the response
series and the initial parameter values used to reproduce the reference result.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from src.loglik_barma import loglik_barma

# Setup the directory
DATA_DIR: Path = Path(__file__).parent.parent / "data" / "raw"


def test_loglikelihood_barma():
    """Ensure the Python log-likelihood function evaluated at the initial value
    matches from the reference value from R"""

    # ---------------------------------------------------------------------------------
    # 1. LOAD VALUES FROM R
    # ---------------------------------------------------------------------------------

    loglik_df: pd.DataFrame = pd.read_csv(DATA_DIR / "loglik_value_data.csv")

    y_raw: pd.DataFrame = pd.read_csv(DATA_DIR / "y_train_data.csv")
    y: pd.Series = pd.Series(y_raw["x"])

    iv_raw: pd.DataFrame = pd.read_csv(DATA_DIR / "start_values_data.csv")
    iv: pd.Series = iv_raw.set_index("param")["x"]

    X_train: np.ndarray = pd.read_csv(DATA_DIR / "X_train_data.csv")[
        ["hs", "hc"]
    ].to_numpy()

    LOG_LIKELIHOOD_VALUE_FROM_R = loglik_df["value"].iloc[0]

    # ---------------------------------------------------------------------------------
    # 2. COMPUTE THE LOG-LIKELIHOOD USING PYTHON
    # ---------------------------------------------------------------------------------

    ar = [10, 18]
    ma = [1, 13]

    LOG_LIKELIHOOD_VALUE_USING_PYTHON = loglik_barma(
        y=y,
        ar=ar,
        ma=ma,
        alpha=iv["alpha"],
        varphi=iv[["varphi10", "varphi18"]],
        theta=iv[["theta1", "theta13"]],
        beta=iv[["hs", "hc"]],
        phi=iv["phi"],
        exog=X_train,
    )

    # ---------------------------------------------------------------------------------
    # 3. ASSERT THE MATCH
    # ---------------------------------------------------------------------------------

    np.testing.assert_allclose(
        LOG_LIKELIHOOD_VALUE_FROM_R,
        LOG_LIKELIHOOD_VALUE_USING_PYTHON,
        rtol=1e-10,
        atol=1e-10,
    )
