"""BARMAX model score vector calculation module.

This module provides functions for computing score vector (gradient of the
log-likelihood) values for Beta AutoRegressive Moving Average with eXogenous
regressors (BARMAX) models.
"""

import pandas as pd
import numpy as np
import numpy.typing as npt
from scipy.special import digamma
from src.make_link_structure import make_link_structure
from src.utils import _validate_estimation_inputs, _build_model_config

__all__ = ["score_vector_barma"]


def score_vector_barma(
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
) -> np.ndarray:
    """Calculate the score vector for a Beta AutoRegressive Moving Average
    model with eXogenous regressors.

    Computes the analytic gradient of the conditional log-likelihood of a
    BARMAX model with respect to all parameters. The model combines AR and MA
    components on a transformed scale, with exogenous regressors entering the
    linear predictor. The first ``max(ar_order, ma_order)`` observations are
    used to initialise the recursion and are excluded from the score.

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
    np.ndarray of shape (1 + p + q + k + 1,)
        Score vector ordered as
        ``(alpha, varphi_1, ..., varphi_p, theta_1, ..., theta_q,
        beta_1, ..., beta_k, phi)``,
        where ``p = len(ar_lags)``, ``q = len(ma_lags)`` and
        ``k = len(beta)``.

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
    Evaluate the score vector of a BARMAX(1, 1) model with one exogenous
    regressor at a given parameter vector:

    >>> import numpy as np
    >>> import pandas as pd
    >>> y = pd.Series([0.2, 0.4, 0.6, 0.5, 0.3, 0.4, 0.3, 0.5])
    >>> exog = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    >>> score_vector_barma(
    ...     y, ar_lags=[1], ma_lags=[1],
    ...     alpha=0.0, varphi=0.3, theta=0.1,
    ...     exog=exog, beta=0.05,
    ...     phi=20.0, link="logit",
    ... )  # doctest: +SKIP
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
    # 2. MODEL CONFIGURATION
    # ---------------------------------------------------------------------------------
    (
        max_lag,
        names_varphi,
        names_theta,
        names_beta,
        n_ar_params,
        n_ma_params,
        n_beta_params,
    ) = _build_model_config(ar_lags=ar_lags, ma_lags=ma_lags, exog=exog)

    # Load link functions based on the specified link type
    linkfun, linkinv, mu_eta = make_link_structure(link)

    # Transform the y variable using the link function
    y_transformed = linkfun(y)

    n_obs = len(y)

    # ---------------------------------------------------------------------------------
    # 3. CALCULATE ERROR AND PREDICTOR
    # ---------------------------------------------------------------------------------
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
    #    n_params = 1 + n_ar_params + n_ma_params + n_beta_params + 1
    #    return np.zeros(n_params) + 1e10

    # Calculate linear predictor (eta) and error for each observation
    for t in range(max_lag, n_obs):
        ar_exog_term = (
            np.dot(varphi, y_transformed[t - ar_lags] - xb[t - ar_lags])
            if n_ar_params > 0
            else 0.0
        )
        ma_term = np.dot(theta, error[t - ma_lags]) if n_ma_params > 0 else 0.0
        eta[t] = alpha + xb[t] + ar_exog_term + ma_term
        error[t] = y_transformed[t] - eta[t]

    # Extract effective observations
    idx_effective = np.arange(max_lag, n_obs)
    eta_eff = eta[idx_effective]
    y_eff = y[idx_effective]

    # Transform linear predictor to mean scale using inverse link function
    mu_eff = linkinv(eta=eta_eff)
    mu_eff = np.clip(mu_eff, 1e-10, 1 - 1e-10)

    # ---------------------------------------------------------------------------------
    # 4. COMPUTE DERIVATIVES OF ETA W.R.T. PARAMETERS
    # ---------------------------------------------------------------------------------
    # This section calculates the partial derivatives of eta_t w.r.t. each parameter
    # using recursive relationships for ARMA models

    # Design matrices for parameter derivatives
    # P_exog[t]: derivatives of AR terms w.r.t. varphi
    # R[t]: derivatives of MA terms w.r.t. theta
    # M_exog[t]: derivatives of exog terms w.r.t. beta
    P_exog = np.zeros((n_obs - max_lag, n_ar_params))
    R = np.zeros((n_obs - max_lag, n_ma_params))
    M_exog = np.zeros((n_obs - max_lag, n_beta_params))

    for t in range(n_obs - max_lag):
        P_exog[t] = y_transformed[t + max_lag - ar_lags] - np.dot(
            exog[t + max_lag - ar_lags], beta
        )

        R[t] = error[t + max_lag - ma_lags]

        # x_t minus its AR-weighted lags, for every regressor column at once.
        # exog[t + max_lag - ar_lags] has shape (n_ar_params, n_exog_params);
        # varphi @ (.) contracts the lag axis -> shape (n_exog_params,).
        M_exog[t] = exog[t + max_lag] - varphi @ exog[t + max_lag - ar_lags]

    # Initialize derivative arrays
    d_eta_d_alpha = np.zeros((n_obs))
    d_eta_d_varphi = np.zeros((n_obs, n_ar_params))
    d_eta_d_theta = np.zeros((n_obs, n_ma_params))
    d_eta_d_beta = np.zeros((n_obs, n_beta_params))

    # Recursive computation of derivatives
    for t in range(max_lag, n_obs):
        d_eta_d_alpha[t] = 1 - theta @ d_eta_d_alpha[t - ma_lags]
        d_eta_d_varphi[t] = P_exog[t - max_lag, :] - theta @ d_eta_d_varphi[t - ma_lags]
        d_eta_d_theta[t] = R[t - max_lag, :] - theta @ d_eta_d_theta[t - ma_lags]
        d_eta_d_beta[t] = M_exog[t - max_lag, :] - theta @ d_eta_d_beta[t - ma_lags]

    # Subset to effective sample
    s = d_eta_d_alpha[idx_effective]
    rP = d_eta_d_varphi[idx_effective]
    rR = d_eta_d_theta[idx_effective]
    rM_exog = d_eta_d_beta[idx_effective]

    # ---------------------------------------------------------------------------------
    # 5. ASSEMBLE THE SCORE VECTOR
    # ---------------------------------------------------------------------------------
    # Derivative of the inverse link w.r.t. eta (i.e. d(mu)/d(eta)).
    mu_eta_values = mu_eta(eta_eff)

    ystar = np.log(y_eff / (1 - y_eff))
    mustar = digamma(mu_eff * phi) - digamma((1 - mu_eff) * phi)

    ystar_mustar = ystar - mustar
    mT_ystar_mustar = mu_eta_values * ystar_mustar

    # Score components for each parameter block.
    U_alpha = phi * np.dot(s, mT_ystar_mustar)
    U_varphi = phi * np.dot(rP.T, mT_ystar_mustar)
    U_theta = phi * np.dot(rR.T, mT_ystar_mustar)
    U_beta = phi * np.dot(rM_exog.T, mT_ystar_mustar)
    U_phi = np.sum(
        mu_eff * ystar_mustar
        + np.log(1 - y_eff)
        - digamma((1 - mu_eff) * phi)
        + digamma(phi)
    )

    # Concatenate into final score vector:
    # (alpha, varphi..., theta..., beta..., phi)
    escore_vec = np.concatenate(
        [np.array([U_alpha]), U_varphi, U_theta, U_beta, np.array([U_phi])]
    )

    if np.any(np.isnan(escore_vec)):
        return np.zeros_like(escore_vec) + 1e10

    return escore_vec
