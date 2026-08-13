import numpy as np
import pandas as pd

# Explicitly declare this module has no public API
__all__ = []


# =====================================================================================
# Auxiliary for the helper function
# =====================================================================================
def _validate_response(y):
    # Validate time series object
    if not isinstance(y, pd.Series):
        raise TypeError("'y' must be a time series")

    # Check for missing values
    if y.isna().any():
        raise ValueError("'y' contains missing values (NaN).")

    # Check unit interval bounds
    if not np.all((y > 0) & (y < 1)):
        raise ValueError(
            f"'y' values must be strictly in (0, 1). "
            f"Current range: [{np.min(y):.4f}, {np.max(y):.4f}]"
        )

    y = np.asarray(y, dtype=float)

    return y


def _validate_lags(lags, component):

    # Validate
    if lags is None:
        return np.array([], dtype=int)

    # Coerce
    lags = np.atleast_1d(np.asarray(lags, dtype=int))
    if lags.size > 0 and np.any(lags <= 0):
        raise ValueError(f"{component} lags must be strictly positive.")
    return lags


def _validate_exog(exog, y):
    # Safely handle the exogenous variable (it might be None)
    if exog is not None:
        # -----------------------------------------------------------------------------
        # 1. Validade the exog
        # -----------------------------------------------------------------------------
        # Standardize numpy array
        exog = np.asarray(exog, dtype=float)

        if exog.shape[0] != len(y):
            raise ValueError(
                f"'exog' must have {len(y)} rows to match length of 'y'. "
                f"Current: {exog.shape[0]}"
            )

        # -----------------------------------------------------------------------------
        # 2. Coerce the exog
        # -----------------------------------------------------------------------------
        if exog.ndim == 1:
            exog = exog.reshape(-1, 1)  # safe only applies to 1D input
    else:
        # if exog was None, return matrix column of zeros
        exog = np.zeros((len(y), 0))

    return exog


def _validate_estimates(y, ar_lags, ma_lags, alpha, varphi, theta, beta, phi, exog):

    exog = _validate_exog(exog=exog, y=y)

    # Coerce estimates
    alpha = float(alpha)
    phi = float(phi)

    varphi = (
        np.zeros(0)
        if len(ar_lags) == 0
        else np.asarray(varphi, dtype=float).reshape(-1)
    )

    theta = (
        np.zeros(0) if len(ma_lags) == 0 else np.asarray(theta, dtype=float).reshape(-1)
    )

    beta = (
        np.zeros(0) if exog.shape[1] == 0 else np.asarray(beta, dtype=float).reshape(-1)
    )

    return y, alpha, varphi, theta, beta, phi, exog


# =====================================================================================
# Helper function
# =====================================================================================
def _validate_estimation_inputs(
    y=None,
    ar=None,
    ma=None,
    alpha=None,
    varphi=None,
    theta=None,
    beta=None,
    phi=None,
    exog=None,
):
    """Validate and sanitize inputs for log-likelihood and score vector functions.

    Ensures the response variable is a valid pandas Series strictly bounded
    in (0, 1), safely converts all lag and coefficient structures to NumPy
    arrays, and verifies matrix dimensional alignment for exogenous variables.

    Called by loglik_barma() and score_vector_barma(), which receive explicit
    parameter values. For the estimation function barma(), use _validate_y()
    instead, since parameter values are not available prior to optimization.

    Parameters
    ----------
    y : pd.Series
        Response variable time series, bounded strictly in (0, 1).
    ar : array-like or None
        AR lag indices. If None, no AR component is included.
    ma : array-like or None
        MA lag indices. If None, no MA component is included.
    alpha : float
        Intercept of the linear predictor.
    varphi : array-like or float
        AR coefficients corresponding to ar lags.
    theta : array-like or float
        MA coefficients corresponding to ma lags.
    beta : array-like or float
        Coefficients for the exogenous regressors.
    phi : float
        Precision parameter of the Beta distribution. Must be positive.
    exog : np.ndarray of shape (n_obs, k)
        Validated exogenous matrix. k = 0 (zero-width) when no regressors
        are provided. Never None after validation.

    Returns
    -------
    y : pd.Series
        Validated response variable, unchanged.
    ar_lags : np.ndarray of dtype int
        AR lags as a 1D integer array, or empty array if ar is None.
    ma_lags : np.ndarray of dtype int
        MA lags as a 1D integer array, or empty array if ma is None.
    alpha : float
    varphi : np.ndarray of dtype float
        AR coefficients as a 1D float array.
    theta : np.ndarray of dtype float
        MA coefficients as a 1D float array.
    beta : np.ndarray of dtype float
        Exogenous coefficients as a 1D float array.
    phi : float
    exog : np.ndarray of shape (n_obs, k)
        Coerced exogenous matrix.

    Raises
    ------
    TypeError
        If y is not a pd.Series.
    ValueError
        If y contains NaN values.
    ValueError
        If y values are not strictly in (0, 1).
    ValueError
        If any AR or MA lag is not strictly positive.
    ValueError
        If exog row count does not match len(y).
    ValueError
        If exog column count does not match len(beta).
    """

    # ---------------------------------------------------------------------------------
    # 1. Validate input values
    # ---------------------------------------------------------------------------------
    y = np.array(y, dtype=float)

    # ---------------------------------------------------------------------------------
    # 2. Convert inputs to numpy arrays
    # ---------------------------------------------------------------------------------
    # Safely handle AR and MA lags (they might be None)
    ar_lags = _validate_lags(ar, "AR")
    ma_lags = _validate_lags(ma, "MA")

    y, alpha, varphi, theta, beta, phi, exog = _validate_estimates(
        y=y,
        ar_lags=ar_lags,
        ma_lags=ma_lags,
        alpha=alpha,
        varphi=varphi,
        theta=theta,
        beta=beta,
        phi=phi,
        exog=exog,
    )

    return y, ar_lags, ma_lags, alpha, varphi, theta, beta, phi, exog


def _validate_model_specification(y, ar, ma, exog):
    """Validate y and coerce ar/ma/exog for the estimation function.

    Validates the response variable, lag specifications, and exogenous
    regressors. Returns coerced numpy arrays ready for use in barma().
    Unlike _validate_estimation_inputs(), does not require parameter values
    (alpha, varphi, theta, beta, phi), which are estimated, not supplied.

    Parameters
    ----------
    y : pd.Series
        Response variable time series, bounded strictly in (0, 1).
    ar : array-like or None
        AR lag indices. If None, no AR component is included.
    ma : array-like or None
        MA lag indices. If None, no MA component is included.
    exog : array-like or None
        Matrix of exogenous regressors of shape (n_obs, k). A 1D array is
        treated as a single regressor and reshaped to (n_obs, 1) internally.

    Returns
    -------
    y : pd.Series
        Validated response variable, unchanged.
    ar_lags : np.ndarray of dtype int
        AR lags as a 1D integer array, or empty array if ar is None.
    ma_lags : np.ndarray of dtype int
        MA lags as a 1D integer array, or empty array if ma is None.
    exog : np.ndarray of shape (n_obs, k)
        Validated exogenous matrix. k = 0 (zero-width) when no regressors
        are provided. Never None after validation.

    Raises
    ------
    TypeError
        If y is not a pd.Series.
    ValueError
        If y contains NaN values.
    ValueError
        If y values are not strictly in (0, 1).
    ValueError
        If any AR or MA lag is not strictly positive.
    """

    # ---------------------------------------------------------------------------------
    # 1. Validate the time series
    # ---------------------------------------------------------------------------------
    y = _validate_response(y)

    # ---------------------------------------------------------------------------------
    # 2. Validate the AR/MA inputs
    # ---------------------------------------------------------------------------------
    ar_lags = _validate_lags(ar, "AR")
    ma_lags = _validate_lags(ma, "MA")

    # ---------------------------------------------------------------------------------
    # 3. Validate the exogenous
    # ---------------------------------------------------------------------------------
    exog = _validate_exog(exog=exog, y=y)

    return y, ar_lags, ma_lags, exog


def _build_model_config(ar_lags, ma_lags, exog):
    """Build parameter names, counts, and lag orders from validated inputs.

    Derives all structural quantities needed to index into the parameter
    vector and construct the linear predictor. Receives already-validated
    and coerced inputs from _validate_estimation_inputs() or
    _validate_estimation_inputs().

    Parameters
    ----------
    ar_lags : np.ndarray of dtype int
        Validated AR lag indices, as returned by _validate_model_specification().
        Empty array if no AR component is present.
    ma_lags : np.ndarray of dtype int
        Validated MA lag indices, as returned by _validate_model_specification().
        Empty array if no MA component is present.
    exog : np.ndarray of shape (n_obs, k)
        Validated exogenous matrix. k = 0 (zero-width) when no regressors
        are provided. Never None after validation.

    Returns
    -------
    max_lag : int
        Maximum lag across AR and MA components. Used to determine the
        number of initial observations lost to conditioning.
    names_varphi : list of str
        Parameter names for AR coefficients, e.g. ['varphi10', 'varphi18'].
        Empty list if no AR component.
    names_theta : list of str
        Parameter names for MA coefficients, e.g. ['theta1', 'theta13'].
        Empty list if no MA component.
    names_beta : list of str
        Parameter names for exogenous coefficients, e.g. ['beta1', 'beta2'].
        Empty list if exog is None.
    n_ar_params : int
        Number of AR parameters.
    n_ma_params : int
        Number of MA parameters.
    n_beta_params : int
        Number of exogenous parameters.
    """

    # ---------------------------------------------------------------------------------
    # 1. AR component
    # ---------------------------------------------------------------------------------
    if len(ar_lags) > 0:
        names_varphi = [f"varphi{lag}" for lag in ar_lags]
        n_ar_params = len(ar_lags)
        ar_order = np.max(ar_lags)
    else:
        names_varphi = []
        n_ar_params = 0
        ar_order = 0

    # ---------------------------------------------------------------------------------
    # 2. MA component
    # ---------------------------------------------------------------------------------
    if len(ma_lags) > 0:
        names_theta = [f"theta{lag}" for lag in ma_lags]
        n_ma_params = len(ma_lags)
        ma_order = np.max(ma_lags)
    else:
        names_theta = []
        n_ma_params = 0
        ma_order = 0

    # ---------------------------------------------------------------------------------
    # 3. Exogenous regressors
    # ---------------------------------------------------------------------------------
    n_beta_params = exog.shape[1]
    names_beta = [f"beta{i + 1}" for i in range(n_beta_params)]

    max_lag = max(ar_order, ma_order)

    return (
        max_lag,
        names_varphi,
        names_theta,
        names_beta,
        n_ar_params,
        n_ma_params,
        n_beta_params,
    )
