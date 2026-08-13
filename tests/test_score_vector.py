"""Unit tests for the BARMA score-vector calculation.

This module validates the `score-vector_barma` implementation against a reference
value computed in R, ensuring the Python port reproduces the expected score-
vector for a fixed dataset and set of parameters.

The tests rely on CSV fixtures in `data/raw/` that contain both the response
series and the initial parameter values used to reproduce the reference result.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from src.score_vector_barma import score_vector_barma

# Setup the directory
DATA_DIR: Path = Path(__file__).parent.parent / "data" / "raw"


def test_score_vector_barma_barma():
    """Start values for BARMA model, special case with AR, MA and Regressors.
    Ensure the Python start values function evaluated at the time series
    matches from the reference value from R"""

    # ---------------------------------------------------------------------------------
    # 1. LOAD VALUES FROM R
    # ---------------------------------------------------------------------------------

    reference_data: pd.DataFrame = pd.read_csv(DATA_DIR / "reference_data.csv")

    y_raw: pd.DataFrame = pd.read_csv(DATA_DIR / "y_train_data.csv")
    y: pd.Series = pd.Series(y_raw["x"])

    iv_raw: pd.DataFrame = pd.read_csv(DATA_DIR / "start_values_data.csv")
    iv: pd.Series = iv_raw.set_index("param")["x"]

    X_train: np.ndarray = pd.read_csv(DATA_DIR / "X_train_data.csv")[
        ["hs", "hc"]
    ].to_numpy()

    SCORE_VECTOR_VALUE_FROM_R = reference_data["score_value"]

    # ---------------------------------------------------------------------------------
    # 2. COMPUTE THE SCORE VECTOR VALUE USING PYTHON
    # ---------------------------------------------------------------------------------

    ar = [10, 18]
    ma = [1, 13]

    SCORE_VECTOR_VALUE_USING_PYTHON = score_vector_barma(
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
        SCORE_VECTOR_VALUE_FROM_R,
        SCORE_VECTOR_VALUE_USING_PYTHON,
        rtol=1e-10,
        atol=1e-10,
    )
