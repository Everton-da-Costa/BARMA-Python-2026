import numpy as np
import pandas as pd

from src.make_link_structure import make_link_structure
from src.utils import _build_model_config


def fitted_barma(results_obj) -> pd.Series:
    """
    Calculate the fitted values for a fitted BARMA model.

    Computes the in-sample fitted values (conditional means) based on the
    estimated parameters, AR/MA components, and exogenous regressors.

    Parameters
    ----------
    fit : dict
        A dictionary containing the fitted model results, typically returned
        by the `barma` optimization function. Must contain:
        - "y": Response variable time series (pd.Series).
        - "ar_lags": Lags for the AR component (list or None).
        - "ma_lags": Lags for the MA component (list or None).
        - "exog": Exogenous regressors matrix (array-like or None).
        - "link": Name of the link function (str).
        - "estimates": Estimated parameters (pd.Series).

    Returns
    -------
    pd.Series
        A pandas Series of the fitted values on the mean scale, matching
        the index of the original response variable `y`. The first `max_lag`
        observations will be NaN.
    """

    # ---------------------------------------------------------------------------------
    # 1. Load the link function and model configuration
    # ---------------------------------------------------------------------------------
    y = results_obj.model.y
    ar_lags = results_obj.model.ar_lags
    ma_lags = results_obj.model.ma_lags
    exog = results_obj.model.exog
    link = results_obj.model.link

    estimates = results_obj.estimates

    alpha = estimates["alpha"]

    # Load link function
    linkfun, linkinv, _ = make_link_structure(link)

    n_obs = len(y)
    y_transformed = linkfun(y).values

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

    # Extract estimated coefficients
    varphi = estimates[names_varphi].values if n_ar_params > 0 else np.array([])
    theta = estimates[names_theta].values if n_ma_params > 0 else np.array([])
    beta = estimates[names_beta].values if n_beta_params > 0 else np.array([])

    xb = exog @ beta if n_beta_params > 0 else np.zeros(n_obs)

    # ---------------------------------------------------------------------------------
    # 3. CALCULATE ERROR AND PREDICTOR
    # ---------------------------------------------------------------------------------
    # eta: predictor scale
    # error: y_transformed - eta
    eta = np.full(n_obs, np.nan)
    error = np.zeros(n_obs)

    for t in range(max_lag, n_obs):
        ar_exog_term = (
            np.dot(varphi, y_transformed[t - ar_lags] - xb[t - ar_lags])
            if n_ar_params > 0
            else 0.0
        )
        ma_term = np.dot(theta, error[t - ma_lags]) if n_ma_params > 0 else 0.0
        eta[t] = alpha + xb[t] + ar_exog_term + ma_term
        error[t] = y_transformed[t] - eta[t]

    # ---------------------------------------------------------------------------------
    # 4. Output
    # ---------------------------------------------------------------------------------
    eta_eff = eta[max_lag:]

    # Transform linear predictor to mean scale using inverse link function
    mu_eff = linkinv(eta_eff)

    fitted_array = np.concatenate((np.full(max_lag, np.nan), mu_eff))

    fitted = pd.Series(fitted_array, index=y.index, name="Fitted_Values")

    return fitted
