"""BARMAX model log-likelihood calculation module.

This module provides functions for computing log-likelihood values
for Beta AutoRegressive Moving Average with eXogenous regressors
(BARMAX) models.
"""

import pandas as pd
import numpy as np
import numpy.typing as npt
from scipy.stats import beta as beta_dist
from src.make_link_structure import make_link_structure
from src.utils import _validate_estimation_inputs

__all__ = ["loglik_barma"]


def loglik_barma(
    y: pd.Series,
    ar: npt.ArrayLike | None = None,
    ma: npt.ArrayLike | None = None,
    alpha: float = 0.0,
    varphi: npt.ArrayLike | float = 0.0,
    theta: npt.ArrayLike | float = 0.0,
    beta: npt.ArrayLike | float = 0.0,
    phi: float = 20.0,
    exog: npt.ArrayLike | None = None,
    link: str = "logit",
) -> float:
    """Calculate log-likelihood for a Beta AutoRegressive Moving Average model
    with exogenous regressors.

    Computes the conditional log-likelihood for a BARMAX model with a
    specified link function. The model combines AR and MA components on
    a transformed scale, with exogenous regressors entering the linear predictor.
    The first ``max(ar_order, ma_order)`` observations are used to initialise
    the recursion and are excluded from the likelihood sum.

    Parameters
    ----------
    y : pd.Series
        Response variable time series, bounded strictly in (0, 1).
        Must not contain missing values.
    ar : npt.ArrayLike or None, optional
        Lags for the autoregressive (AR) component. If array-like, specifies
        the exact lags to include. If None, no AR component is included.
        Default is None.
    ma : npt.ArrayLike or None, optional
        Lags for the moving average (MA) component. If array-like, specifies
        the exact lags to include. If None, no MA component is included.
        Default is None.
    alpha : float, optional
        Intercept term of the linear predictor. Default is 0.0.
    varphi : npt.ArrayLike or float, optional
        AR coefficients corresponding to ``ar_lags``. Default is 0.0.
    theta : npt.ArrayLike or float, optional
        MA coefficients corresponding to ``ma_lags``. Default is 0.0.
    beta : npt.ArrayLike or float, optional
        Coefficients for the exogenous regressors in ``exog``. Default is 0.0.
    phi : float, optional
        Precision (dispersion) parameter of the Beta distribution.
        Must be positive. Default is 20.0.
    exog : npt.ArrayLike or None, optional
        Matrix of exogenous regressors of shape (n_obs, k). A 1D array is
        treated as a single regressor and reshaped to (n_obs, 1) internally.
        Default is None.
    link : str, optional
        Link function name (e.g., ``"logit"``). Must be supported by
        :func:`make_link_structure`. Default is "logit".

    Returns
    -------
    float
        Sum of conditional log-likelihood values across all effective
        observations (i.e., excluding the initialisation period).

    Raises
    ------
    TypeError
        If ``y`` is not a time series.
    ValueError
        If ``y`` contains missing values (NaN).
    ValueError
        If ``y`` values are not strictly within the (0, 1) interval.
    ValueError
        If ``link`` is not a supported link function name.
    ValueError
        If ``exog`` has a different number of rows than ``y``.
    ValueError
        If the number of columns in ``exog`` does not match the size of ``beta``.

    Examples
    --------
    Evaluate the log-likelihood of a BARMAX(1, 1) model with one
    exogenous regressor at a given parameter vector:

    >>> import numpy as np
    >>> import pandas as pd
    >>> y = pd.Series([0.2, 0.4, 0.6, 0.5, 0.3, 0.4, 0.3, 0.5])
    >>> exog = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    >>> loglik_barma(
    ...     y, ar_lags=[1], ma_lags=[1],
    ...     alpha=0.0, varphi=0.3, theta=0.1,
    ...     exog=exog, beta=0.05,
    ...     phi=20.0, link="logit",
    ... )
    4.978697100211956
    """
    # ---------------------------------------------------------------------------------
    # 1. VALIDATE INPUT PARAMETERS
    # ---------------------------------------------------------------------------------
    (y, ar_lags, ma_lags, alpha, varphi, theta, beta, phi, exog) = (
        _validate_estimation_inputs(
            y=y,
            ar=ar,
            ma=ma,
            alpha=alpha,
            varphi=varphi,
            theta=theta,
            beta=beta,
            phi=phi,
            exog=exog,
        )
    )

    # ---------------------------------------------------------------------------------
    # 2. SETUP LINK FUNCTIONS AND TIME SERIES PROPERTIES
    # ---------------------------------------------------------------------------------

    # Load link functions based on the specified link type
    linkfun, linkinv, _ = make_link_structure(link)

    # Transform the y variable using the link function
    y_transformed = linkfun(y)

    n_obs = len(y)

    # ---------------------------------------------------------------------------------
    # 3. CALCULATE ERROR AND PREDICTOR
    # ---------------------------------------------------------------------------------

    ## Determine model orders and effective sample size
    # Safely calculate max lags (default to 0 if the array is empty)
    ar_order = int(np.max(ar_lags)) if len(ar_lags) > 0 else 0
    ma_order = int(np.max(ma_lags)) if len(ma_lags) > 0 else 0
    max_lag = max(ar_order, ma_order)

    ## Initialize arrays for recursive computations
    # eta: predictor scale
    # error: y_transformed - eta
    eta = np.full(n_obs, np.nan)
    error = np.zeros(n_obs)

    if exog.size == []:
        xb = np.zeros(n_obs)
    else:
        # Pre-compute X * beta for efficiency (vectorized operation)
        xb = np.dot(exog, beta)

    # if phi <= 0:
    #    return -1e10

    # Calculate linear predictor (eta) and error for each observation
    for t in range(max_lag, n_obs):
        ar_exog_term = np.dot(varphi, y_transformed[t - ar_lags] - xb[t - ar_lags])

        ma_term = np.dot(theta, error[t - ma_lags])

        eta[t] = alpha + xb[t] + ar_exog_term + ma_term
        error[t] = y_transformed[t] - eta[t]

    # ---------------------------------------------------------------------------------
    # 4. CALCULATE THE FINAL LOG-LIKELIHOOD
    # ----------------------------------------------------------------------------------

    # Extract effective observations
    idx_effective = np.arange(max_lag, n_obs)
    eta_eff = eta[idx_effective]
    y_eff = y[idx_effective]

    # Transform linear predictor to mean scale using inverse link function
    mu_eff = linkinv(eta=eta_eff)
    mu_eff = np.clip(mu_eff, 1e-10, 1 - 1e-10)

    # Calculate Beta PDF values for each observation
    ll_terms = beta_dist.logpdf(y_eff, a=mu_eff * phi, b=(1 - mu_eff) * phi)

    if np.any(np.isnan(ll_terms)):
        return -1e10

    # Return sum of log-likelihood
    return float(np.sum(ll_terms))
