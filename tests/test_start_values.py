"""Unit tests for the BARMA start values calculation.

This module validates the `start_values` implementation against a reference
value computed in R, ensuring the Python port reproduces the start values.

The tests rely on CSV fixtures in `data/raw/` that contain the response
series used to reproduce the reference result.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from src.start_values import start_values

# Setup the directory
DATA_DIR: Path = Path(__file__).parent.parent / "data" / "raw"


def test_start_values():

    # ---------------------------------------------------------------------------------
    # 1. LOAD VALUES FROM R
    # ---------------------------------------------------------------------------------

    y_raw: pd.DataFrame = pd.read_csv(DATA_DIR / "y_train_data.csv")
    y: pd.Series = pd.Series(y_raw["x"])

    iv_raw: pd.DataFrame = pd.read_csv(DATA_DIR / "start_values_data.csv")
    iv: pd.Series = iv_raw.set_index("param")["x"]

    X_train: np.ndarray = pd.read_csv(DATA_DIR / "X_train_data.csv")[
        ["hs", "hc"]
    ].to_numpy()

    START_VALUES_FROM_R = iv

    # ---------------------------------------------------------------------------------
    # 2. COMPUTE THE START VALUES USING PYTHON
    # ---------------------------------------------------------------------------------
    ar = [10, 18]
    ma = [1, 13]

    START_VALUES_USING_PYTHON = start_values(
        y=y, ar=ar, ma=ma, exog=X_train, link="logit"
    )

    # ---------------------------------------------------------------------------------
    # 3. ASSERT THE MATCH
    # ---------------------------------------------------------------------------------

    np.testing.assert_allclose(
        START_VALUES_FROM_R,
        START_VALUES_USING_PYTHON,
        rtol=1e-10,
        atol=1e-10,
    )
