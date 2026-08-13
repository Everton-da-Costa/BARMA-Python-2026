"""Unit tests for the BARMA start values calculation.

This module validates the `start_values` implementation against a reference
value computed in R, ensuring the Python port reproduces the start values.

The tests rely on CSV fixtures in `data/raw/` that contain the response
series used to reproduce the reference result.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from src.model import BARMA

# Setup the directory
DATA_DIR: Path = Path(__file__).parent.parent / "data" / "raw"


def test_barma():

    # ---------------------------------------------------------------------------------
    # 1. LOAD VALUES FROM R
    # ---------------------------------------------------------------------------------
    reference_data: pd.DataFrame = pd.read_csv(DATA_DIR / "reference_data.csv")

    y_raw: pd.DataFrame = pd.read_csv(DATA_DIR / "y_train_data.csv")
    y: pd.Series = pd.Series(y_raw["x"])

    X_train: np.ndarray = pd.read_csv(DATA_DIR / "X_train_data.csv")[
        ["hs", "hc"]
    ].to_numpy()

    ESTIAMTES_BARMA_FROM_R = reference_data["estimates"]

    # ---------------------------------------------------------------------------------
    # 2. COMPUTE THE ESTIMATES VALUES USING PYTHON
    # ---------------------------------------------------------------------------------

    ar = [10, 18]
    ma = [1, 13]

    BARMA_MODEL_USING_PYTHON = BARMA(y=y, ar=ar, ma=ma, exog=X_train, link="logit")
    BARMA_FIT_USING_PYTHON = BARMA_MODEL_USING_PYTHON.fit()
    BARMA_FIT_USING_PYTHON_ESTIMATES = BARMA_FIT_USING_PYTHON.estimates

    # ---------------------------------------------------------------------------------
    # 3. ASSERT THE MATCH
    # ---------------------------------------------------------------------------------

    np.testing.assert_allclose(
        ESTIAMTES_BARMA_FROM_R,
        BARMA_FIT_USING_PYTHON_ESTIMATES,
        rtol=1e-3,
        atol=0,
    )
