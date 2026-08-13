import pandas as pd
import numpy as np
import numpy.typing as npt

from src.make_link_structure import make_link_structure


def start_values(
    y: pd.Series,
    ar: npt.ArrayLike | None = None,
    ma: npt.ArrayLike | None = None,
    exog: npt.ArrayLike | None = None,
    link: str = "logit",
) -> np.ndarray:
    """Compute starting parameter values for BARMA(X) model estimation.

    Produces initial values for the maximum-likelihood optimizer via a single
    OLS fit on the link-transformed response. The AR coefficients and the
    intercept are estimated jointly with any exogenous regressors. The MA
    coefficients are initialized to zero (MA innovations are latent and cannot
    be recovered by OLS on the response). The precision parameter phi is
    estimated from the OLS residuals using the mean-precision parameterization
    of Ferrari & Cribari-Neto (2004).

    Parameters
    ----------
    y : pd.Series
        Response variable, strictly bounded in (0, 1) and free of NaN.
    ar : array-like or None, optional
        Lags for the autoregressive component (e.g. [1, 12]). None or empty
        omits the AR component. Default is None.
    ma : array-like or None, optional
        Lags for the moving average component. None or empty omits the MA
        component. Default is None.
    exog : array-like of shape (n_obs, k) or None, optional
        Matrix of exogenous regressors. Default is None.
    link : str, optional
        Link function name. Must be supported by make_link_structure.
        Default is "logit".

    Returns
    -------
    pd.Series
        Named series of starting values in the order
        (alpha, varphi..., theta..., beta..., phi).

    References
    ----------
    Ferrari, S.L.P. & Cribari-Neto, F. (2004). Beta regression for modelling
    rates and proportions. Journal of Applied Statistics, 31(7), 799-815.

    Rocha, A.V. & Cribari-Neto, F. (2009). Beta autoregressive moving average
    models. TEST, 18(3), 529-545.
    """

    # ---------------------------------------------------------------------------------
    # 1. VALIDATE INPUT PARAMETERS
    # ---------------------------------------------------------------------------------
    # The validation of the input values and model configuration is already done in
    # the following steps of the model.py file:
    ## # 1. VALIDATE INPUT PARAMETERS
    ## # 2. Model Configuration

    # ---------------------------------------------------------------------------------
    # 2. SETUP LINK FUNCTIONS AND TRANSFORM THE RESPONSE
    # ---------------------------------------------------------------------------------
    linkfun, linkinv, mu_eta = make_link_structure(link)
    y = np.asarray(y, dtype=float)
    y_transformed = linkfun(y)

    # ---------------------------------------------------------------------------------
    # 3. MODEL CONFIGURATION
    # ---------------------------------------------------------------------------------
    # Coerce AR/MA specifications into a consistent shape; empty when absent.
    ar = [] if ar is None else ar
    ma = [] if ma is None else ma

    if len(ar) > 0:
        ar_lags = ar
        ar_order = max(ar_lags)
        n_ar_params = len(ar)
        names_varphi = [f"varphi{lag}" for lag in ar_lags]
    else:
        ar_lags = []
        ar_order = 0
        n_ar_params = 0
        names_varphi = []

    if len(ma) > 0:
        ma_lags = ma
        ma_order = max(ma_lags)
        n_ma_params = len(ma)
        names_theta = [f"theta{lag}" for lag in ma_lags]
    else:
        ma_lags = []
        ma_order = 0
        n_ma_params = len(ma)
        names_theta = []

    n_exog = 0 if exog is None else exog.shape[1]
    names_beta = [f"beta{i + 1}" for i in range(n_exog)]

    # Effective sample excludes the initialization period.
    max_lag = max(ar_order, ma_order)
    n_obs = len(y)
    n_eff = n_obs - max_lag

    # ---------------------------------------------------------------------------------
    # 4. BUILD THE OLS DESIGN MATRIX AND FIT
    # ---------------------------------------------------------------------------------
    # Start values come from one OLS fit on the transformed scale.
    # Coefficient layout in fit_coef: [alpha, varphi_1..p, beta_1..k_exog].
    P_mat = y_transformed[np.arange(n_eff)[:, None] + max_lag - ar_lags]
    x_inter = np.ones((n_eff, 1))
    y_ols = y_transformed[max_lag:]

    if exog is not None:
        x_exog = exog[max_lag:n_obs]
        x_ols = np.hstack([x_inter, P_mat, x_exog])
    else:
        x_ols = np.hstack([x_inter, P_mat])

    fit_coef, _, _, _ = np.linalg.lstsq(x_ols, y_ols, rcond=None)

    # ---------------------------------------------------------------------------------
    # 5. UNPACK COEFFICIENTS
    # ---------------------------------------------------------------------------------
    # Layout: (alpha, varphi, theta, beta, phi). theta is zero by construction;
    # MA errors are latent and cannot be estimated by OLS on the response.
    alpha_start = fit_coef[0]
    varphi_start = fit_coef[1 : 1 + n_ar_params]
    theta_start = np.zeros(n_ma_params)
    beta_start = fit_coef[1 + n_ar_params :]

    # ---------------------------------------------------------------------------------
    # 6. ESTIMATE phi FROM THE OLS RESIDUALS
    # ---------------------------------------------------------------------------------
    k = len(fit_coef)
    eta_hat_ols = x_ols @ fit_coef
    mu_ols = linkinv(eta_hat_ols)
    resid_ols = y_ols - eta_hat_ols

    # d(mu)/d(eta) converts variability between scales (Ferrari & Cribari-Neto,
    # 2004, eq. 2.4); the reciprocal appears in sigma2 below.
    linkfun_deriv = 1.0 / mu_eta(eta_hat_ols)

    sigma2 = np.sum(resid_ols**2) / ((n_eff - k) * linkfun_deriv**2)
    phi_start_aux = np.sum(mu_ols * (1 - mu_ols) / sigma2)
    phi_start = phi_start_aux / n_eff

    # ---------------------------------------------------------------------------------
    # 7. OUTPUT: INITIAL VALUES
    # ---------------------------------------------------------------------------------

    initial_values = np.concatenate(
        [
            np.array([alpha_start]),
            varphi_start,
            theta_start,
            beta_start,
            np.array([phi_start]),
        ]
    )

    names_start = ["alpha"] + names_varphi + names_theta + names_beta + ["phi"]
    initial_values = pd.Series(initial_values, index=names_start)

    print(initial_values)
    # >>> print(initial_values)
    #    alpha        0.681132
    #    varphi10    -0.189152
    #    varphi18     0.367029
    #    theta1       0.000000
    #    theta13      0.000000
    #    beta1        0.656474
    #    beta2        1.198639
    #    phi         47.293020
    #    dtype: float64

    return initial_values
