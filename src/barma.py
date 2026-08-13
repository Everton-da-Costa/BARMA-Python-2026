import numpy as np
import pandas as pd
import numpy.typing as npt

from src.start_values import start_values
from src.loglik_barma import loglik_barma
from src.score_vector_barma import score_vector_barma

from src.utils import _validate_inputs, _build_model_config

from scipy.optimize import minimize

__all__ = ["barma"]


def barma(
    y: pd.Series,
    ar: npt.ArrayLike | None = None,
    ma: npt.ArrayLike | None = None,
    exog: npt.ArrayLike | None = None,
    link: str = "logit",
) -> dict:
    """Calculate the estimates of Beta AutoRegressive Moving Average
    model with eXogenous regressors.
    Computes the optimization procedure using the conditional log-likelihood  and
    score verctor of a BARMAX model with respect to all parameters.
    The model combines AR and MA
    components on a transformed scale, with exogenous regressors entering the
    linear predictor.
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
    dict
        A dictionary containing:
        - "estimates"      : pd.Series of estimated parameters
        - "converged"      : bool, optimizer convergence status
        - "log-likelihood" : float, maximized log-likelihood
        - "AIC"            : float, Akaike information criterion
        - "BIC"            : float, Bayesian information criterion
        - "n_iter"        : int, number of optimizer iterations
        - "opt_details"    : OptimizeResult, full scipy optimizer output
        - "y"              : pd.Series, original response variable
        - "ar"             : list, AR lags
        - "ma"             : list, MA lags
        - "exog"           : array-like or None, exogenous regressors
        - "link"           : str, link function name

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
    Do it at the end
    """

    # ---------------------------------------------------------------------------------
    # 1. VALIDATE INPUT PARAMETERS
    # ---------------------------------------------------------------------------------
    (y, ar_lags, ma_lags, exog) = _validate_inputs(y=y, ar=ar, ma=ma, exog=exog)

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
        n_beta,
    ) = _build_model_config(ar_lags=ar_lags, ma_lags=ma_lags, exog=exog)

    # ---------------------------------------------------------------------------------
    # 3. DEFINITION OF THE NEGATIVE FUNCTIONS
    # ---------------------------------------------------------------------------------
    def neg_loglik_barma(params, y, ar_lags, ma_lags, exog, link):
        loglik_barma_values = loglik_barma(
            y=y,
            ar=ar_lags,
            ma=ma_lags,
            alpha=params[0],
            varphi=params[1 : 1 + n_ar_params],
            theta=params[1 + n_ar_params : 1 + n_ar_params + n_ma_params],
            beta=params[
                1 + n_ar_params + n_ma_params : 1 + n_ar_params + n_ma_params + n_beta
            ],
            phi=params[1 + n_ar_params + n_ma_params + n_beta],
            exog=exog,
            link=link,
        )

        return -loglik_barma_values

    def neg_score_vector_barma(params, y, ar_lags, ma_lags, exog, link):
        score_vector_barma_values = score_vector_barma(
            y=y,
            ar=ar_lags,
            ma=ma_lags,
            alpha=params[0],
            varphi=params[1 : 1 + n_ar_params],
            theta=params[1 + n_ar_params : 1 + n_ar_params + n_ma_params],
            beta=params[
                1 + n_ar_params + n_ma_params : 1 + n_ar_params + n_ma_params + n_beta
            ],
            phi=params[1 + n_ar_params + n_ma_params + n_beta],
            exog=exog,
            link=link,
        )
        return -score_vector_barma_values

    # ---------------------------------------------------------------------------------
    # 4. OPTIMIZATION PROCEDURE
    # ---------------------------------------------------------------------------------
    initial_values = start_values(y=y, ar=ar_lags, ma=ma_lags, exog=exog, link=link)

    opt_raw = minimize(
        fun=neg_loglik_barma,
        x0=initial_values,
        jac=neg_score_vector_barma,
        method="BFGS",
        args=(y, ar_lags, ma_lags, exog, link),
    )

    names_params = ["alpha"] + names_varphi + names_theta + names_beta + ["phi"]
    estimates = pd.Series(opt_raw.x, index=names_params, name="Estimates")

    k = len(estimates)
    n_obs = len(y)
    log_lik = -opt_raw.fun
    aic = 2 * k - 2 * log_lik
    bic = k * np.log(n_obs) - 2 * log_lik

    output_dic = {
        "estimates": estimates,
        "converged": opt_raw.success,
        "log-likelihood": round(log_lik, 5),
        "AIC": round(aic, 5),
        "BIC": round(bic, 5),
        "n_iter": opt_raw.nit,
        "opt_details": opt_raw,
        "y": y,
        "ar_lags": ar_lags,
        "ma_lags": ma_lags,
        "max_lag": max_lag,
        "names_params": names_params,
        "exog": exog,
        "link": link,
    }

    return output_dic
