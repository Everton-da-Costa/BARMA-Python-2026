# =====================================================================================
# Imports
# =====================================================================================

# -----------------------------------------------------------------------------
# Standard library
# -----------------------------------------------------------------------------
# warning of convergence failure
import warnings
from typing import ClassVar

# -----------------------------------------------------------------------------
# Third-party
# -----------------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf, pacf

# -----------------------------------------------------------------------------
# Local
# -----------------------------------------------------------------------------
from src.loglik_barma import loglik_barma
from src.make_link_structure import make_link_structure
from src.score_vector_barma import score_vector_barma
from src.utils import _build_model_config, _validate_model_specification

plt.style.use("ggplot")

# =====================================================================================
# class BARMA: Optimization procedure
# =====================================================================================


class BARMA:
    """Beta Autoregressive Moving Average model with optional exogenous regressors."""

    def __init__(self, y, ar=None, ma=None, exog=None, link="logit"):

        # -----------------------------------------------------------------------------
        # 1. VALIDATE INPUT PARAMETERS
        # -----------------------------------------------------------------------------
        (self.y, self.ar_lags, self.ma_lags, self.exog) = _validate_model_specification(
            y=y, ar=ar, ma=ma, exog=exog
        )

        # -----------------------------------------------------------------------------
        # 2. Model Configuration
        # -----------------------------------------------------------------------------
        (
            self.max_lag,
            self.names_varphi,
            self.names_theta,
            self.names_beta,
            self.n_ar_params,
            self.n_ma_params,
            self.n_beta_params,
        ) = _build_model_config(
            ar_lags=self.ar_lags, ma_lags=self.ma_lags, exog=self.exog
        )

        self.link = link
        self.y_index = y.index
        self._fim = None

    def fit(self):
        # -----------------------------------------------------------------------------
        # 1. Definition of the negative functions
        # -----------------------------------------------------------------------------
        def neg_loglik_barma(params):
            loglik_barma_values = loglik_barma(
                y=self.y,
                ar=self.ar_lags,
                ma=self.ma_lags,
                alpha=params[0],
                varphi=params[1 : 1 + self.n_ar_params],
                theta=params[
                    1 + self.n_ar_params : 1 + self.n_ar_params + self.n_ma_params
                ],
                beta=params[
                    1 + self.n_ar_params + self.n_ma_params : 1
                    + self.n_ar_params
                    + self.n_ma_params
                    + self.n_beta_params
                ],
                phi=params[
                    1 + self.n_ar_params + self.n_ma_params + self.n_beta_params
                ],
                exog=self.exog,
                link=self.link,
            )

            return -loglik_barma_values

        def neg_score_vector_barma(params):
            score_vector_barma_values = score_vector_barma(
                y=self.y,
                ar=self.ar_lags,
                ma=self.ma_lags,
                alpha=params[0],
                varphi=params[1 : 1 + self.n_ar_params],
                theta=params[
                    1 + self.n_ar_params : 1 + self.n_ar_params + self.n_ma_params
                ],
                beta=params[
                    1 + self.n_ar_params + self.n_ma_params : 1
                    + self.n_ar_params
                    + self.n_ma_params
                    + self.n_beta_params
                ],
                phi=params[
                    1 + self.n_ar_params + self.n_ma_params + self.n_beta_params
                ],
                exog=self.exog,
                link=self.link,
            )
            return -score_vector_barma_values

        # -----------------------------------------------------------------------------
        # 2. Optimization procedure
        # -----------------------------------------------------------------------------
        initial_values = self._compute_start_values()

        opt_raw = minimize(
            fun=neg_loglik_barma,
            x0=initial_values,
            jac=neg_score_vector_barma,
            method="BFGS",
        )

        if not opt_raw.success:
            warnings.warn(
                f"Optimization failed to converge: {opt_raw.message}",
                UserWarning,
                stacklevel=2,
            )

        results = BARMAResults(model=self, opt_raw=opt_raw)

        return results

    def __repr__(self):
        return (
            f"BARMA(ar={self.ar_lags},ma={self.ma_lags}, "
            f"exog={self.has_beta}, link='{self.link}')"
        )

    def _compute_start_values(self):

        # -----------------------------------------------------------------------------
        # 1. Extract the model configuration
        # -----------------------------------------------------------------------------

        link = self.link
        y = self.y

        ar_lags = self.ar_lags
        max_lag = self.max_lag
        exog = self.exog

        n_ar_params = self.n_ar_params
        n_ma_params = self.n_ma_params

        has_ar = self.has_ar
        has_ma = self.has_ma
        has_beta = self.has_beta

        # -----------------------------------------------------------------------------
        # 2. Setup the link functions
        # -----------------------------------------------------------------------------
        linkfun, linkinv, mu_eta = make_link_structure(link)
        y_transformed = linkfun(y)

        # -----------------------------------------------------------------------------
        # 3. Model configuration
        # -----------------------------------------------------------------------------
        n_obs = len(y)
        n_eff = n_obs - self.max_lag

        # Will be overridden in the process if necessary
        varphi0 = np.zeros(self.n_ar_params)
        beta0 = np.zeros(self.n_beta_params)
        # -----------------------------------------------------------------------------
        # 4. Build the OLS design matrix and fit
        # -----------------------------------------------------------------------------
        y_ols = y_transformed[max_lag:]
        mean_y = np.mean(y)

        if not has_ar and has_ma and not has_beta:
            alpha0 = np.mean(y_transformed)
            theta0 = np.zeros(n_ma_params)
            phi0 = (mean_y * (1 - mean_y)) / np.var(y, ddof=1)

        elif not has_ar and has_ma and has_beta:
            x_inter = np.ones((n_eff, 1))
            x_exog_eff = exog[max_lag:]

            x_ols = np.hstack([x_inter, x_exog_eff])

            fit_coef, _, _, _ = np.linalg.lstsq(x_ols, y_ols, rcond=None)

            alpha0 = np.mean(y_transformed)
            theta0 = np.zeros(n_ma_params)
            beta0 = fit_coef[1:]
            phi0 = (mean_y * (1 - mean_y)) / np.var(y, ddof=1)

        else:
            # Start values come from one OLS fit on the transformed scale.
            # Coefficient layout in fit_coef: [alpha, varphi_1..p, beta_1..k_exog].
            P_mat = y_transformed[np.arange(n_eff)[:, None] + max_lag - ar_lags]
            x_inter = np.ones((n_eff, 1))
            y_ols = y_transformed[max_lag:]

            if exog.shape[1] != 0:
                x_exog_eff = exog[max_lag:]
                x_ols = np.hstack([x_inter, P_mat, x_exog_eff])
            else:
                x_ols = np.hstack([x_inter, P_mat])

            fit_coef, _, _, _ = np.linalg.lstsq(x_ols, y_ols, rcond=None)

            # -------------------------------------------------------------------------
            # 5. Unpack the coefficients
            # -------------------------------------------------------------------------
            # Layout: (alpha, varphi, theta, beta, phi). theta is zero by construction;
            # MA errors are latent and cannot be estimated by OLS on the response.
            alpha0 = fit_coef[0]
            varphi0 = fit_coef[1 : 1 + n_ar_params]
            beta0 = fit_coef[1 + n_ar_params :]

            theta0 = np.zeros(n_ma_params)

            # -------------------------------------------------------------------------
            # 6. Estimate of phi from the OLS residuals
            # -------------------------------------------------------------------------
            k = len(fit_coef)
            eta_hat_ols = x_ols @ fit_coef
            mu_ols = linkinv(eta_hat_ols)
            resid_ols = y_ols - eta_hat_ols

            # d(mu)/d(eta) converts variability between scales (Ferrari & Cribari-Neto,
            # 2004, eq. 2.4); the reciprocal appears in sigma2_t below.
            # g'(mu_hat) = deta/dmu = 1 / (dmu/deta) = 1 / mu_eta(eta)
            linkfun_deriv = 1.0 / mu_eta(eta_hat_ols)

            sigma2_t = np.sum(resid_ols**2) / ((n_eff - k) * linkfun_deriv**2)
            phi0_aux = np.sum(mu_ols * (1 - mu_ols) / sigma2_t)
            phi0 = phi0_aux / n_eff

        # -----------------------------------------------------------------------------
        # 7. Initial values
        # -----------------------------------------------------------------------------

        initial_values = np.concatenate(
            [
                np.array([alpha0]),
                varphi0,
                theta0,
                beta0,
                np.array([phi0]),
            ]
        )

        # Uncomment if you will work on the start values
        # names_start = ["alpha"] + names_varphi + names_theta + names_beta + ["phi"]
        # initial_values = pd.Series(initial_values, index=names_start)

        return initial_values

    # ---------------------------------------------------------------------------------
    # Flags to improve readability
    # ---------------------------------------------------------------------------------
    @property
    def has_ar(self):
        return self.n_ar_params > 0

    @property
    def has_ma(self):
        return self.n_ma_params > 0

    @property
    def has_beta(self):
        return self.n_beta_params > 0


# =====================================================================================
# class BARMAResults:
# =====================================================================================
class BARMAResults:
    """Container for fitted BARMA model results."""

    # Definition of the residuals type supported
    _VALID_RESID_TYPES: ClassVar[set[str]] = {"pearson", "raw", "scale"}

    def __init__(self, model, opt_raw):
        self.model = model
        self.opt_raw = opt_raw

        # All lazy caches declared
        self._fim = None
        self._summary = None
        self._fitted = None
        self._residuals_cache = {}
        self._ljungbox_cache = {}

        self._mu_hat = None
        self._eta_hat = None
        self._error_hat = None

        self._forecast_values = None

        names_params = (
            ["alpha"]
            + self.model.names_varphi
            + self.model.names_theta
            + self.model.names_beta
            + ["phi"]
        )

        estimates = pd.Series(opt_raw.x, index=names_params, name="Estimates")

        # -----------------------------------------------------------------------------
        # Present: Estimates and optimization results
        # -----------------------------------------------------------------------------
        self.estimates = estimates
        self.converged = opt_raw.success
        self.n_iter = opt_raw.nit

    def __repr__(self):
        ar_order = self.model.n_ar_params
        ma_order = self.model.n_ma_params
        exog_order = self.model.n_beta_params

        return (
            f"<BARMAResults:"
            f"AR({ar_order}), MA({ma_order}), Exog({exog_order}) | "
            f"Converged: {self.converged} | "
            f"Log-Likelihood: {-self.opt_raw.fun:.3f}>"
        )

    # ---------------------------------------------------------------------------------
    # Properties: Information criteria
    # ---------------------------------------------------------------------------------
    @property
    def log_likelihood(self):
        return float(-self.opt_raw.fun)

    @property
    def aic(self):
        k = len(self.estimates)
        return float(2 * k - 2 * self.log_likelihood)

    @property
    def bic(self):
        n_obs = len(self.model.y)
        k = len(self.estimates)
        return float(k * np.log(n_obs) - 2 * self.log_likelihood)

    # ---------------------------------------------------------------------------------
    # Properties: Model outputs (lazy)
    # ---------------------------------------------------------------------------------
    @property
    def fitted_values(self):
        if self._fitted is None:
            self._compute_fitted_internal()
        return self._fitted

    @property
    def fim_barma(self):
        if self._fim is None:
            self._compute_fim_barma_internal()
        return self._fim

    # =================================================================================
    # Public Methods
    # =================================================================================
    def summary(self):
        if self._summary is None:
            self._compute_summary_internal()
        return self._summary

    def residuals(self, resid_type="pearson"):
        """Returns the requested residuals, computing them lazily if needed."""

        if resid_type not in self._VALID_RESID_TYPES:
            raise ValueError(
                f"resid_type must be one of {self._VALID_RESID_TYPES!r}, "
                f"got {resid_type!r}."
            )

        if resid_type not in self._residuals_cache:
            self._compute_residuals_internal(resid_type)

        return self._residuals_cache[resid_type]

    def forecast(self, h=12, exog=None):
        """Returns the requested forecast, computing them lazily if needed."""

        self._compute_forecast_internal(h=h, exog=exog)

        return self._forecast_values

    # =================================================================================
    # Fisher Information Matrix (FIM)
    # =================================================================================
    def _compute_fim_barma_internal(self):
        from scipy.special import polygamma

        # -----------------------------------------------------------------------------
        # 1. Extract the model configuration and link function
        # -----------------------------------------------------------------------------
        model = self.model

        y = model.y
        n_obs = len(y)

        # Extract the model configuration

        # Lags
        ar_lags = model.ar_lags
        ma_lags = model.ma_lags
        max_lag = model.max_lag

        # Flags for the presence of a determinable component of the model
        has_ar = model.has_ar
        has_ma = model.has_ma

        # Number of AR/MA components in the model
        n_ar_params = model.n_ar_params
        n_ma_params = model.n_ma_params
        n_beta_params = model.n_beta_params

        # Extract the estimates of the model
        names_varphi = model.names_varphi
        names_theta = model.names_theta
        names_beta = model.names_beta

        alpha = self.estimates["alpha"]
        varphi = self.estimates[names_varphi].values
        theta = self.estimates[names_theta].values
        beta = self.estimates[names_beta].values
        phi = self.estimates["phi"]

        # Extract the Link functions
        linkfun, linkinv, mu_eta = make_link_structure(self.model.link)

        # Compute the y transformed and exogenous * beta matrix
        y_transformed = linkfun(y)
        exog = model.exog
        xb = exog @ beta

        # -----------------------------------------------------------------------------
        # 2. Compute error and predictor
        # -----------------------------------------------------------------------------
        # eta: predictor scale
        # error: y_transformed - eta
        eta_hat = np.full(n_obs, np.nan)
        error_hat = np.zeros(n_obs)

        # Calculate linear predictor (eta) and error for each observation
        for t in range(max_lag, n_obs):
            ar_exog_term = (
                np.dot(varphi, y_transformed[t - ar_lags] - xb[t - ar_lags])
                if has_ar
                else 0.0
            )
            ma_term = np.dot(theta, error_hat[t - ma_lags]) if has_ma else 0.0

            eta_hat[t] = alpha + xb[t] + ar_exog_term + ma_term
            error_hat[t] = y_transformed[t] - eta_hat[t]

        # Extract effective observations
        idx_effective = np.arange(max_lag, n_obs)
        eta_eff = eta_hat[idx_effective]

        # Transform linear predictor to mean scale using inverse link function
        mu_eff = linkinv(eta=eta_eff)

        # -----------------------------------------------------------------------------
        # 3. Compute derivatives of eta W.R.T Parameters
        # -----------------------------------------------------------------------------
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

            R[t] = error_hat[t + max_lag - ma_lags]

            # x_t minus its AR-weighted lags, for every regressor column at once.
            # exog[t + max_lag - ar_lags] has shape (n_ar_params, n_exog_params);
            # varphi @ (.) contracts the lag axis -> shape (n_exog_params,).
            M_exog[t] = exog[t + max_lag] - varphi @ exog[t + max_lag - ar_lags]

        # Initialize derivative arrays
        d_eta_d_alpha = np.zeros(n_obs)
        d_eta_d_varphi = np.zeros((n_obs, n_ar_params))
        d_eta_d_theta = np.zeros((n_obs, n_ma_params))
        d_eta_d_beta = np.zeros((n_obs, n_beta_params))

        # Recursive computation of derivatives
        for t in range(max_lag, n_obs):
            d_eta_d_alpha[t] = 1 - theta @ d_eta_d_alpha[t - ma_lags]
            d_eta_d_varphi[t] = (
                P_exog[t - max_lag, :] - theta @ d_eta_d_varphi[t - ma_lags]
            )
            d_eta_d_theta[t] = R[t - max_lag, :] - theta @ d_eta_d_theta[t - ma_lags]
            d_eta_d_beta[t] = M_exog[t - max_lag, :] - theta @ d_eta_d_beta[t - ma_lags]

        # Extract effective derivative vectors/matrices
        s_vec = d_eta_d_alpha[idx_effective]
        P_mat = d_eta_d_varphi[idx_effective]
        R_mat = d_eta_d_theta[idx_effective]
        M_mat = d_eta_d_beta[idx_effective]

        # -----------------------------------------------------------------------------
        # 4. Compute FIM component vectors (efficiently)
        # -----------------------------------------------------------------------------
        # Trigamma functions: derivatives of digamma
        trigamma_p = polygamma(1, mu_eff * phi)
        trigamma_q = polygamma(1, (1 - mu_eff) * phi)

        # d(mu)/d(eta) - derivative of inverse link function
        mu_eta_val = mu_eta(eta=eta_eff)

        # Vector W
        # W = phi * {psi'(mu*phi) + psi'((1-mu)*phi)} * (mu_eta)^2
        w_t_vec = phi * (trigamma_p + trigamma_q) * mu_eta_val**2

        # Vector c
        # c = phi * {psi'(mu*phi)*mu - psi'((1-mu)*phi)*(1-mu)}
        c_t_vec = phi * (trigamma_p * mu_eff - trigamma_q * (1 - mu_eff))

        # Vector d
        # d = psi'(mu*phi)*mu^2 + psi'((1-mu)*phi)*(1-mu)^2 - psi'(phi)
        d_t_vec = (
            trigamma_p * mu_eff**2 + trigamma_q * (1 - mu_eff) ** 2 - polygamma(1, phi)
        )

        # -----------------------------------------------------------------------------
        # 5 Compute FIM Blocs
        # -----------------------------------------------------------------------------
        def _calc_block(X, Y, w, scale=1):
            """Compute scale * X.T @ (Y * w) for FIM block assembly.

            Parameters
            ----------
            X : array-like of shape (n, p) or (n,)
                Left design matrix. 1D arrays are reshaped to column vectors.
            Y : array-like of shape (n, q) or (n,)
                Right design matrix. 1D arrays are reshaped to column vectors.
            w : array-like of shape (n,)
                Weight vector applied row-wise to Y.
            scale : float, optional
                Scalar multiplier applied to the result. Default is 1.

            Returns
            -------
            np.ndarray of shape (p, q)
                The weighted cross-product X.T @ (Y * w) * scale.
                Returns an empty (0, 0) array if X or Y is empty.
            """

            X = np.asarray(X)
            Y = np.asarray(Y)
            w = np.asarray(w).flatten()

            if X.ndim == 1:
                X = X.reshape(-1, 1)

            if Y.ndim == 1:
                Y = Y.reshape(-1, 1)

            if len(w) != Y.shape[0]:
                raise ValueError(
                    "The vector length must match the number of rows in Y."
                )

            Y_weighted = Y * w.reshape(-1, 1)

            result = (X.T @ Y_weighted) * scale

            return result

        # Diagonal matrix
        K_aa = _calc_block(X=s_vec, Y=s_vec, w=w_t_vec, scale=phi)
        K_pp = _calc_block(X=P_mat, Y=P_mat, w=w_t_vec, scale=phi)
        K_tt = _calc_block(X=R_mat, Y=R_mat, w=w_t_vec, scale=phi)
        K_bb = _calc_block(X=M_mat, Y=M_mat, w=w_t_vec, scale=phi)
        K_phiphi = np.sum(d_t_vec)

        # Blocks scaling with phi
        K_ap = _calc_block(X=s_vec, Y=P_mat, w=w_t_vec, scale=phi)
        K_at = _calc_block(X=s_vec, Y=R_mat, w=w_t_vec, scale=phi)
        K_ab = _calc_block(X=s_vec, Y=M_mat, w=w_t_vec, scale=phi)

        K_pt = _calc_block(X=P_mat, Y=R_mat, w=w_t_vec, scale=phi)
        K_pb = _calc_block(X=P_mat, Y=M_mat, w=w_t_vec, scale=phi)

        K_tb = _calc_block(X=R_mat, Y=M_mat, w=w_t_vec, scale=phi)

        # Blocks involving phi
        K_aphi = s_vec @ (mu_eta_val * c_t_vec)
        K_pphi = P_mat.T @ (mu_eta_val * c_t_vec).reshape(-1, 1)
        K_tphi = R_mat.T @ (mu_eta_val * c_t_vec).reshape(-1, 1)
        K_bphi = M_mat.T @ (mu_eta_val * c_t_vec).reshape(-1, 1)

        # --------------------------------------------------------------------------
        # 6. Assemble and return the final FIM
        # --------------------------------------------------------------------------

        # Parameter order: alpha, varphi, theta, beta, phi
        names_params = self.estimates.index
        fim = pd.DataFrame(np.nan, index=names_params, columns=names_params)

        # Diagonal matrix
        fim.loc[["alpha"], ["alpha"]] = K_aa
        fim.loc[names_varphi, names_varphi] = K_pp
        fim.loc[names_theta, names_theta] = K_tt
        fim.loc[names_beta, names_beta] = K_bb
        fim.loc[["phi"], ["phi"]] = K_phiphi

        # Blocks scaling with phi
        fim.loc[["alpha"], names_varphi] = K_ap
        fim.loc[names_varphi, ["alpha"]] = K_ap.T

        fim.loc[["alpha"], names_theta] = K_at
        fim.loc[names_theta, ["alpha"]] = K_at.T

        fim.loc[["alpha"], names_beta] = K_ab
        fim.loc[names_beta, ["alpha"]] = K_ab.T

        fim.loc[names_varphi, names_theta] = K_pt
        fim.loc[names_theta, names_varphi] = K_pt.T

        fim.loc[names_varphi, names_beta] = K_pb
        fim.loc[names_beta, names_varphi] = K_pb.T

        fim.loc[names_theta, names_beta] = K_tb
        fim.loc[names_beta, names_theta] = K_tb.T

        # Blocks involving phi
        fim.loc[["alpha"], ["phi"]] = K_aphi
        fim.loc[["phi"], ["alpha"]] = K_aphi  # scalar

        fim.loc[names_varphi, ["phi"]] = K_pphi
        fim.loc[["phi"], names_varphi] = K_pphi.T

        fim.loc[names_theta, ["phi"]] = K_tphi
        fim.loc[["phi"], names_theta] = K_tphi.T

        fim.loc[names_beta, ["phi"]] = K_bphi
        fim.loc[["phi"], names_beta] = K_bphi.T

        self._fim = fim

        # Export to use in the computation of the forecast and fitted values
        mu_hat = np.concatenate([np.full(max_lag, np.nan), mu_eff])

        self._eta_hat = eta_hat
        self._error_hat = error_hat
        self._mu_hat = mu_hat

    # =================================================================================
    # Fitted values
    # =================================================================================
    def _compute_fitted_internal(self):
        if self._mu_hat is None:
            self._compute_fim_barma_internal()

        self._fitted = pd.Series(
            self._mu_hat, index=self.model.y_index, name="Fitted_Values"
        )

    # =================================================================================
    # Summary of the fit
    # =================================================================================
    def _compute_summary_internal(self):
        from scipy.stats import norm

        # -----------------------------------------------------------------------------
        # 1. Extract the estimates and the Fisher Information Matrixlink function
        # -----------------------------------------------------------------------------
        estimates = self.estimates
        fim = self.fim_barma

        # -----------------------------------------------------------------------------
        # 2. Compute the summary data frame
        # -----------------------------------------------------------------------------
        try:
            vcov_array = np.linalg.inv(fim)
            vcov = pd.DataFrame(vcov_array, index=fim.index, columns=fim.columns)

            # Calculate Std. Errors and p-values
            std_error = np.sqrt(np.diag(vcov))
            zstat = estimates / std_error
            z_pvalues = 2 * (1 - norm.cdf(np.abs(zstat)))

            model_table = {
                "Estimate": estimates,
                "Std. Error": std_error,
                "z value": zstat,
                "Pr(>|z|)": z_pvalues,
            }

            summary_df = pd.DataFrame(model_table)

            self._summary = summary_df

        except np.linalg.LinAlgError:
            self._summary = None
            print(
                "Warning: Fisher Information Matrix is singular and cannot be inverted."
            )

    # =================================================================================
    # Residuals
    # =================================================================================
    def _compute_residuals_internal(self, resid_type):
        # -----------------------------------------------------------------------------
        # 1. Extract model configuration and compute prerequisites
        # -----------------------------------------------------------------------------
        fitted_values = self.fitted_values

        y = self.model.y
        phi = self.estimates["phi"]
        max_lag = self.model.max_lag
        n_obs = len(y)
        eta_hat = self._eta_hat

        # Extract the Link functions
        linkfun, _, mu_eta = make_link_structure(self.model.link)

        # Compute the y transformed and exogenous * beta matrix
        y_transformed = linkfun(y)

        # ---------------------------------------------------------------------------
        # 2. Restrict to effective observations (drop the first max_lag NAs)
        # ---------------------------------------------------------------------------
        idx_effective = np.arange(max_lag, n_obs)
        mu_hat = np.asarray(fitted_values)

        y_eff = y[idx_effective]
        mu_hat_eff = mu_hat[idx_effective]
        eta_hat_eff = eta_hat[idx_effective]
        y_transformed_eff = y_transformed[idx_effective]

        # ---------------------------------------------------------------------------
        # 3. Compute residuals by type
        # ---------------------------------------------------------------------------
        if resid_type == "pearson":
            # Pearson residuals on the response scale.
            # Reference: Ferrari & Cribari-Neto (2004), adapted to the betaARMA setting.
            # r_t = (y_t - mu_hat_t) / sqrt( mu_hat_t*(1 - mu_hat_t) / (1 + phi) )

            resids = (y_eff - mu_hat_eff) / np.sqrt(
                mu_hat_eff * (1 - mu_hat_eff) / (1 + phi)
            )

        elif resid_type == "raw":
            # Raw residuals on the response scale: y_t - mu_hat_t
            resids = y_eff - mu_hat_eff

        elif resid_type == "scale":
            # Link-scale residuals: (g(y_t) - eta_hat_t) / sd(r_t) on predictor scale.
            # Retained for compatibility with earlier versions of the package.

            # Numerator: g(y_t) - eta_hat_t
            numerator = y_transformed_eff - eta_hat_eff

            # Denominator: sqrt( [g'(mu_hat_t)]^2 * mu_hat_t*(1-mu_hat_t) / (1+phi) )
            # where g'(mu) = deta/dmu = 1 / (dmu/deta)
            dmu_deta_sq = mu_eta(eta_hat_eff) ** 2
            Vmu_hat = mu_hat_eff * (1 - mu_hat_eff)

            denominator = np.sqrt((1 / dmu_deta_sq) * Vmu_hat / (1 + phi))

            resids = numerator / denominator

        # ---------------------------------------------------------------------------
        # 4. Pad the first max_lag positions with NA and return as ts object
        # ---------------------------------------------------------------------------
        final_resids = np.concatenate([np.full(max_lag, np.nan), resids])

        final_resids_ts = pd.Series(
            final_resids, index=self.model.y_index, name="Residuals"
        )

        self._residuals_cache[resid_type] = final_resids_ts

    # =================================================================================
    # Forecast
    # =================================================================================

    # ---------------------------------------------------------------------------------
    # Forecast values
    # ---------------------------------------------------------------------------------
    def _compute_forecast_internal(self, h: int, exog=None):
        """Generates h-step ahead out-of-sample forecasts."""

        if self._eta_hat is None:
            self._compute_fim_barma_internal()

        # -----------------------------------------------------------------------------
        # 1. Extract the model estimates and link function
        # -----------------------------------------------------------------------------
        model = self.model

        y = model.y
        n_obs = len(y)

        alpha = self.estimates["alpha"]
        varphi = self.estimates[model.names_varphi].values
        theta = self.estimates[model.names_theta].values
        beta = self.estimates[model.names_beta].values

        linkfun, linkinv, _ = make_link_structure(self.model.link)

        # -----------------------------------------------------------------------------
        # 2. Extract historical data
        # -----------------------------------------------------------------------------
        y_transformed_hist = linkfun(y)
        eta_hist = self._eta_hat
        error_hist = self._error_hat

        # -----------------------------------------------------------------------------
        # 3. Setup and padded arrays
        # -----------------------------------------------------------------------------
        eta_padded = np.concatenate([eta_hist, np.full(h, np.nan)])
        error_padded = np.concatenate([error_hist, np.zeros(h)])
        y_transformed_padded = np.concatenate([y_transformed_hist, np.zeros(h)])

        # -----------------------------------------------------------------------------
        # 4. Handle Exogenous Regressors
        # -----------------------------------------------------------------------------
        if model.n_beta_params > 0:
            # Model has regressors. Check if the user forgot them.
            if exog is None:
                raise ValueError(
                    "The model was fitted with exogenous variables."
                    "You must to provide an 'exog' array for forecasting."
                )

            xb_hist = model.exog @ beta
            xb_test = exog @ beta
            exog_padded = np.concatenate([xb_hist, xb_test])
        else:
            # Pure ARMA model. Skip the error and safely pad with zeros.
            exog_padded = np.zeros(n_obs + h)

        # -----------------------------------------------------------------------------
        # 5. Compute the forecast
        # -----------------------------------------------------------------------------
        for t in range(n_obs, n_obs + h):
            ar_exog_term = (
                np.dot(
                    varphi,
                    y_transformed_padded[t - model.ar_lags]
                    - exog_padded[t - model.ar_lags],
                )
                if model.n_ar_params > 0
                else 0.0
            )

            ma_term = (
                np.dot(theta, error_padded[t - model.ma_lags])
                if model.n_ma_params > 0
                else 0.0
            )
            eta_padded[t] = alpha + exog_padded[t] + ar_exog_term + ma_term

            # Future expectations
            y_transformed_padded[t] = eta_padded[t]
            error_padded[t] = 0.0

        forecast_values_mu = linkinv(eta_padded[n_obs:])

        # -----------------------------------------------------------------------------
        # 6. Output
        # -----------------------------------------------------------------------------

        # Useful in plot_forecast() function
        self.h = h

        # Forecast values with index
        if exog is not None and hasattr(exog, "index"):
            self._forecast_values = pd.Series(
                forecast_values_mu, index=exog.index, name="Forecast"
            )
        else:
            self._forecast_values = pd.Series(forecast_values_mu, name="Forecast")

    # ---------------------------------------------------------------------------------
    # Forecast plot
    # ---------------------------------------------------------------------------------
    def plot_forecast(self, y_test: pd.Series, exog=None) -> plt.Figure:
        """Plot the observed test data against the forecast values."""

        h = len(y_test)
        if self._forecast_values is None or self.h != h:
            self._compute_forecast_internal(h=h, exog=exog)

        # -----------------------------------------------------------------------------
        # 1. Plot configuration
        # -----------------------------------------------------------------------------
        fig, axes = plt.subplots(figsize=(6, 4))

        # Create a vector for index label
        if isinstance(y_test, pd.Series):
            forecast_index = y_test.index
        else:
            forecast_index = np.arange(1, len(y_test) + 1)

        forecast = pd.Series(self._forecast_values.values, index=forecast_index)

        # -----------------------------------------------------------------------------
        # 2. Plotting
        # -----------------------------------------------------------------------------
        axes.scatter(
            y_test.index,
            y_test.values,
            color="steelblue",
            linewidth=1.0,
            label="Observed",
        )
        axes.plot(
            forecast.index,
            forecast.values,
            color="salmon",
            linewidth=1.0,
            label="Forecast",
        )

        axes.set_title(" ")
        axes.set_xlabel("Time")

        # axes.set_yticks(np.arange(0, 1.1, step=0.10))

        # Legend
        axes.legend(loc="best")
        fig.tight_layout()

        return fig

    # =================================================================================
    # Plot diagnostics
    # =================================================================================
    def plot_diagnostics(self, resid_type="pearson"):

        if resid_type not in self._residuals_cache:
            self._compute_residuals_internal(resid_type)

        # -----------------------------------------------------------------------------
        # 1. Extract the model estimates and link function
        # -----------------------------------------------------------------------------
        y = self.model.y
        y_index = self.model.y_index
        max_lag = self.model.max_lag

        residuals = self.residuals(resid_type=resid_type)
        fitted_values = self.fitted_values

        # -----------------------------------------------------------------------------
        # 2. Plot configuration
        # -----------------------------------------------------------------------------
        fig, axes = plt.subplots(2, 2, figsize=(6.5, 5.5))

        # plt.subplots(rows, cols) returns:
        #   fig   → whole figure object
        #   axes  → 2x2 numpy array of Axes objects
        #
        # Layout:
        #   axes[0,0]  axes[0,1]
        #   axes[1,0]  axes[1,1]

        # -----------------------------------------------------------------------------
        # Panel [0,0]: Observed vs Fitted
        # -----------------------------------------------------------------------------
        axes[0, 0].plot(y_index, y, color="steelblue", linewidth=0.8, label="Observed")
        axes[0, 0].plot(
            fitted_values.index,
            fitted_values.values,
            color="salmon",
            linewidth=0.8,
            label="Fitted",
        )

        axes[0, 0].set_title("Observed vs fitted", fontsize=11)
        axes[0, 0].set_xlabel("Time", fontsize=8)
        axes[0, 0].set_ylabel("Value", fontsize=8)

        axes[0, 0].tick_params(labelsize=8)

        # -----------------------------------------------------------------------------
        # Panel [0,1]: Residuals over Time
        # -----------------------------------------------------------------------------
        axes[0, 1].plot(
            residuals.index,
            residuals.values,
            color="olivedrab",
            linewidth=0.8,
        )

        residuals_eff = residuals.iloc[max_lag:]

        # Aux values for setting the limits of the plot
        min_tick = np.abs(np.min(residuals_eff))
        max_tick = np.abs(np.max(residuals_eff))
        tick_lim = np.ceil(max(min_tick, max_tick))

        axes[0, 1].set_title(
            f"Residuals over time, {resid_type.capitalize()} residuals.", fontsize=11
        )
        axes[0, 1].set_xlabel("Time", fontsize=8)
        axes[0, 1].set_ylabel("Residuals", fontsize=8)

        axes[0, 1].tick_params(labelsize=8)
        axes[0, 1].axhline(y=-3, linewidth=0.8, linestyle="dashed", color="blue")
        axes[0, 1].axhline(y=0, linewidth=0.5, linestyle="-", color="black")
        axes[0, 1].axhline(y=3, linewidth=0.8, linestyle="dashed", color="blue")

        axes[0, 1].set_yticks(np.arange(-tick_lim, tick_lim + 1, step=1))

        # -----------------------------------------------------------------------------
        # Panel [1,0]: Residual ACF
        # -----------------------------------------------------------------------------
        plot_acf(residuals_eff, lags=24, ax=axes[1, 0], zero=False, alpha=None)

        # Aux values for setting the limits of the plot
        ci_bound = 1.96 / np.sqrt(len(residuals_eff))
        acf_values = acf(residuals_eff, nlags=24)
        acf_values_plot = acf_values[1:]
        acf_values_plot_limits = np.max(np.abs(acf_values_plot)) + 0.05

        axes[1, 0].set_title(
            f"Residual ACF, {resid_type.capitalize()} residuals.", fontsize=11
        )
        axes[1, 0].set_xlabel("Lag", fontsize=8)
        axes[1, 0].set_ylabel("ACF", fontsize=8)

        axes[1, 0].axhline(y=ci_bound, color="blue", linestyle="dashed", linewidth=0.8)
        axes[1, 0].axhline(y=-ci_bound, color="blue", linestyle="dashed", linewidth=0.8)

        axes[1, 0].tick_params(labelsize=8)
        axes[1, 0].set_ylim(-acf_values_plot_limits, acf_values_plot_limits)
        axes[1, 0].set_xticks(np.arange(0, 25, step=6))

        # -----------------------------------------------------------------------------
        # Panel [1,1]: Residual PACF
        # -----------------------------------------------------------------------------
        plot_pacf(
            residuals_eff,
            lags=24,
            ax=axes[1, 1],
            zero=False,
            alpha=None,
        )

        # Aux values for setting the limits of the plot
        pacf_values = pacf(residuals_eff, nlags=24)
        pacf_values_plot = pacf_values[1:]
        pacf_values_plot_limits = np.max(np.abs(pacf_values_plot)) + 0.05

        axes[1, 1].set_title(
            f"Residual PACF, {resid_type.capitalize()} residuals.", fontsize=11
        )
        axes[1, 1].set_xlabel("Lags", fontsize=8)
        axes[1, 1].set_ylabel("PACF", fontsize=8)

        axes[1, 1].axhline(ci_bound, color="blue", linestyle="dashed", linewidth=0.8)
        axes[1, 1].axhline(-ci_bound, color="blue", linestyle="dashed", linewidth=0.8)

        axes[1, 1].tick_params(labelsize=8)
        axes[1, 1].set_ylim(-pacf_values_plot_limits, pacf_values_plot_limits)
        axes[1, 1].set_xticks(np.arange(0, 25, step=6))

        fig.tight_layout(h_pad=2.0, w_pad=2.0)

        # acf_values_plot, pacf_values, acf_values_plot_limits, pacf_values_plot_limits

        # ---------------------------------------------------------------------------- #
        # Highlight spikes
        # ---------------------------------------------------------------------------- #
        # Uncomment this block if you want to highlight the spikes
        # abs(spikes) >= CI bound  in the ACF and PACF plots

        # acf_values_plot_df = pd.DataFrame(acf_values_plot)
        # acf_values_plot_problem_boll = np.abs(acf_values_plot) >= ci_bound

        # acf_values_plot_problem = acf_values_plot_df[acf_values_plot_problem_boll]

        # pacf_values_plot_df = pd.DataFrame(pacf_values_plot)
        # pacf_values_plot_problem_boll = np.abs(pacf_values_plot) >= ci_bound

        # pacf_values_plot_problem = pacf_values_plot_df[pacf_values_plot_problem_boll]

        # print(f"CI bound \pm: {ci_bound.round(4)} \n")
        # print(f"ACF: {acf_values_plot.round(4)} \n")
        # print(f"PACF: {pacf_values_plot.round(4)} \n")

        # print("Highlight spikes: \n")
        # print(f"ACF lag: {np.array(acf_values_plot_problem.index + 1).round(4)} \n")
        # print(f"ACF value: {np.array(acf_values_plot_problem.values).round(4)} \n")

        # print(f"PACF lag: {np.array(pacf_values_plot_problem.index + 1).round(4)} \n")
        # print(f"PACF value: {np.array(pacf_values_plot_problem.values).round(4)} \n")

        # spike_acf = np.abs(acf_values_plot) > ci_bound
        # spike_pacf = np.abs(pacf_values_plot) > ci_bound

        # spike_correlation = (spike_acf | spike_pacf).any()

        # return fig, spike_correlation

        # ---------------------------------------------------------------------------- #

        return fig

    # =================================================================================
    # Ljung-Box test
    # =================================================================================

    # ---------------------------------------------------------------------------------
    # Ljung-Box test: stats and p-values
    # ---------------------------------------------------------------------------------
    def ljungbox_test(self, resid_type="pearson"):

        if resid_type not in self._residuals_cache:
            self._compute_residuals_internal(resid_type)

        # To inspect the seasonality we set n = 24
        n_lags = 24
        fit_df = len(self.model.ar_lags) + len(self.model.ma_lags)
        residuals = self.residuals(resid_type=resid_type)
        residuals_eff = residuals.iloc[self.model.max_lag :]

        ljungbox_raw_df = acorr_ljungbox(residuals_eff, lags=n_lags, model_df=fit_df)

        ljungbox_df = ljungbox_raw_df.dropna()

        self._ljungbox_cache[resid_type] = ljungbox_df

        return ljungbox_df

    # ---------------------------------------------------------------------------------
    # Plot lags p-values
    # ---------------------------------------------------------------------------------
    def plot_ljungbox(self, resid_type="pearson"):

        if resid_type not in self._ljungbox_cache:
            self.ljungbox_test(resid_type)

        # To inspect the seasonality we set n = 24
        ljungbox_df = self._ljungbox_cache[resid_type]

        fig, axes = plt.subplots(1, 1, figsize=(6.5, 4.5))

        axes.scatter(
            x=ljungbox_df.index,
            y=ljungbox_df.lb_pvalue,
            linewidth=0.8,
            color="olivedrab",
        )

        axes.set_title(
            f"Ljung-Box p-values, {resid_type.capitalize()} residuals.", pad=20
        )
        axes.set_xlabel("Lag")
        axes.set_ylabel("p-value")

        tick_lim = np.ceil(np.max(np.abs(ljungbox_df.lb_pvalue)) * 10) / 10

        axes.set_xticks(np.arange(0, 26, step=2))
        axes.set_yticks(np.arange(0, tick_lim + 0.05, step=0.05))

        # axes.tick_params(labelsize=8)

        axes.axhline(0.05, color="gray", linestyle="dashed")

        return fig
