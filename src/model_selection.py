"""Stepwise-forward order selection for BARMA (Beta ARMA) models.

Provides :func:`search_component` (the per-component worker) and
:func:`auto_barma` (the stage driver). See :func:`auto_barma` for the full
description of the three-stage search and the optional ``sig_level`` filter.
"""

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

from src.model import BARMA

from pmdarima.arima import auto_arima

from IPython.display import display, Markdown
import matplotlib.pyplot as plt
# from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import acorr_ljungbox


def search_component(
    y: pd.Series,
    exog: npt.ArrayLike | None = None,
    component: str | None = None,
    start_lag: int = 1,
    end_lag: int = 12,
    sig_level: float = 1.00,
    ar_lags_fixed: list | None = None,
    ma_lags_fixed: list | None = None,
    ar_lags_all: list | None = None,
    ma_lags_all: list | None = None,
):
    """Search one component (AR or MA) for significant, converging lags.

    Auxiliary routine for :func:`auto_barma`. Given the component to vary and,
    optionally, the lags held fixed for the *other* component, it fits one BARMA
    model per candidate lag and keeps those that (a) converge and (b) have every
    estimated coefficient significant at ``sig_level``.

    The candidate lags depend on which ``*_fixed`` / ``*_all`` arguments are
    supplied:

    - both ``*_fixed`` are None: the range ``start_lag .. end_lag`` is tried,
      with the other component empty (a pure BAR or BMA model).
    - only the other component fixed: every lag in the matching ``*_all`` pool
      is tried, added on top of the fixed component.
    - both components fixed: the pool for ``component`` minus the lags already
      fixed for it is tried, added on top of both fixed components.

    Parameters
    ----------
    y : pd.Series
        Response variable time series, bounded strictly in (0, 1).
        Must not contain missing values.
    exog : array-like of shape (n_obs, k) or None, optional
        Matrix of exogenous regressors. A 1-D array is treated as a single
        regressor and reshaped to (n_obs, 1) internally. Default is None.
    component : {"AR", "MA"}
        Which component to vary in this call.
    start_lag : int, default 1
        First lag of the search range. Used only for pure models (barma0).
    end_lag : int, default 12
        Last lag of the search range. Used only for pure models (barma0). With
        strong seasonality in monthly data, consider extending it, e.g. 24, 36, 48.
    sig_level : float, default 1.0
        Significance threshold applied to *every* estimated coefficient. The
        default 1.0 keeps all converging models (selection by information
        criterion only, as in auto.arima). A value such as 0.05 additionally
        discards any model with a coefficient whose p-value exceeds it -- note
        this tests every coefficient (intercept, exogenous betas and the
        precision phi included), not just the lag under evaluation.
    ar_lags_fixed : list or None, optional
        AR lags held fixed while searching. Must be None when ``component`` is
        "AR" (a fixed component cannot be searched again). Default is None.
    ma_lags_fixed : list or None, optional
        MA lags held fixed while searching. Must be None when ``component`` is
        "MA". Default is None.
    ar_lags_all : list or None, optional
        Pool of significant AR lags to draw candidates from (from barma0).
        Default is None.
    ma_lags_all : list or None, optional
        Pool of significant MA lags to draw candidates from (from barma0).
        Default is None.

    Returns
    -------
    grid_fit : pd.DataFrame
        One row per accepted model, with columns
        ["AR", "MA", "Conv", "AIC", "BIC"].
    sig_lags : list of int
        The candidate lags whose models were accepted (converged and fully
        significant). This becomes the ``*_lags_all`` pool in later stages.

    Raises
    ------
    ValueError
        If ``component`` is not "AR" or "MA", or if the component requested is
        already fixed (which would mean searching it twice).
    """

    # if the model has both AR and MA components
    search_arma = ar_lags_fixed is not None and ma_lags_fixed is not None

    # if the model has an AR (fixed), then the search will be in the MA component
    search_ma = ar_lags_fixed is not None and ma_lags_fixed is None

    # if the model has an MA (fixed), then the search will be in the AR component
    search_ar = ar_lags_fixed is None and ma_lags_fixed is not None

    # if the model has only an AR or MA component
    barma0 = ar_lags_fixed is None and ma_lags_fixed is None

    # ---------------------------------------------------------------------------------
    #  Validation
    # ---------------------------------------------------------------------------------
    if component not in ("AR", "MA"):
        raise ValueError(f"Please use AR or MA instead of {component}.")

    if component == "AR" and search_ma:
        raise ValueError(
            f"AR is already fixed, ({ar_lags_fixed}), can't search AR again. "
            "To search AR, call again with ar_lags_fixed=None."
        )

    if component == "MA" and search_ar:
        raise ValueError(
            f"MA is already fixed, ({ma_lags_fixed}), can't search MA again. "
            "To search MA, call again with ma_lags_fixed=None."
        )

    # ---------------------------------------------------------------------------------
    # Candidate lags: define the model
    # ---------------------------------------------------------------------------------
    if component == "AR":
        if search_arma:
            candidate_lags = sorted(list(set(ar_lags_all) - set(ar_lags_fixed)))
            ar_base, ma_base = ar_lags_fixed, ma_lags_fixed

        elif search_ar:
            candidate_lags = ar_lags_all
            ar_base, ma_base = [], ma_lags_fixed

        elif barma0:
            candidate_lags = np.arange(start_lag, end_lag + 1, step=1)
            ar_base, ma_base = [], []

    elif component == "MA":
        if search_arma:
            candidate_lags = sorted(list(set(ma_lags_all) - set(ma_lags_fixed)))
            ar_base, ma_base = ar_lags_fixed, ma_lags_fixed

        elif search_ma:
            candidate_lags = ma_lags_all
            ar_base, ma_base = ar_lags_fixed, []

        elif barma0:
            candidate_lags = np.arange(start_lag, end_lag + 1, step=1)
            ar_base, ma_base = [], []

    # ---------------------------------------------------------------------------------
    # Candidate lags: search stage
    # ---------------------------------------------------------------------------------
    grid_fit = []
    sig_lags = []

    for lag_temp in np.array(candidate_lags):
        if component == "AR":
            ar_lags = sorted(ar_base + [int(lag_temp)])
            ma_lags = ma_base

        elif component == "MA":
            ar_lags = ar_base
            ma_lags = sorted(ma_base + [int(lag_temp)])

        barma_model = BARMA(
            y=y,
            ar=ar_lags,
            ma=ma_lags,
            exog=exog,
        )

        try:
            barma_fit = barma_model.fit()
            summary = barma_fit.summary()
            has_sig_estimates = bool((summary["Pr(>|z|)"] <= sig_level).all())

            if barma_fit.converged and has_sig_estimates:
                sig_lags.append(int(lag_temp))
                grid_fit.append(
                    {
                        "AR": ar_lags,
                        "MA": ma_lags,
                        "Conv": barma_fit.converged,
                        "AIC": barma_fit.aic,
                        "BIC": barma_fit.bic,
                    }
                )
        except Exception:
            continue

    grid_fit = pd.DataFrame(grid_fit, columns=["AR", "MA", "Conv", "AIC", "BIC"])

    return grid_fit, sig_lags


def auto_barma(
    y: pd.Series,
    exog: npt.ArrayLike | None = None,
    start_lag: int = 1,
    end_lag: int = 12,
    sig_level: float = 1.00,
    ar_lags_fixed: list | None = None,
    ma_lags_fixed: list | None = None,
    ar_lags_all: list | None = None,
    ma_lags_all: list | None = None,
    search_stage: str = "barma0",
):
    """Run one stage of a stepwise-forward BARMA order search.

    ``auto_barma`` selects the AR/MA lag structure of a BARMA model by
    minimizing the BIC, following the stepwise-forward idea of
    ``forecast::auto.arima`` (Hyndman & Khandakar, 2008). Unlike ``auto.arima``,
    model selection here is by information criterion only -- there is no
    coefficient-significance step unless you set ``sig_level`` below 1.0.

    The search is driven one stage at a time by the caller, so each stage's fitted
    models can be inspected (summaries, diagnostics) before committing to the next. The
    stage index equals the number of lag components already fixed in the base model
    when the stage begins:

        "barma0"  0 fixed  -- fit pure BAR(k) and BMA(k) for k in [start_lag, end_lag]
                                and keep the lags that are individually significant.
        "barma1"  1 fixed  -- fix the best lag of one component and search the other,
                                giving models with one AR lag and one MA lag.
        "barma2"  2 fixed  -- keep both best lags fixed and add one further lag
                                to each.

    Optionally, sig_level filters models by coefficient significance: the default 1.0
    keeps every converging model (BIC decides alone), while a value such as 0.05
    discards any model with an insignificant coefficient. Because each stage is a
    separate call, it may differ between stages-- e.g. 1.0 in barma0 so no lag is
    discarded early, then tighter in barma2 so the final models report only significant
    terms.

    Parameters
    ----------
    y : pd.Series
        Response variable time series, bounded strictly in (0, 1).
        Must not contain missing values.
    exog : array-like of shape (n_obs, k) or None, optional
        Matrix of exogenous regressors. A 1-D array is treated as a single
        regressor and reshaped to (n_obs, 1) internally. Default is None.
    start_lag : int, default 1
        First lag of the pure-model search range (barma0 only).
    end_lag : int, default 12
        Last lag of the pure-model search range (barma0 only). With strong
        seasonality in monthly data consider e.g. 6, 12, 24, 36, 48.
    sig_level : float, default 1.0
        Significance threshold forwarded to :func:`search_component`. The
        default 1.0 applies no significance filter (IC-only selection); a value
        such as 0.05 keeps only models whose coefficients are all significant.
    ar_lags_fixed : list or None, optional
        Best AR lag(s) carried over from the previous stage. Default is None.
    ma_lags_fixed : list or None, optional
        Best MA lag(s) carried over from the previous stage. Default is None.
    ar_lags_all : list or None, optional
        Pool of individually significant AR lags found in barma0. Default None.
    ma_lags_all : list or None, optional
        Pool of individually significant MA lags found in barma0. Default None.
    search_stage : {"barma0", "barma1", "barma2"}, default "barma0"
        Which stage to run (see the summary above).

    Returns
    -------
    For ``search_stage="barma0"`` -- a 4-tuple:
        ar_lags_all : list of int
            Individually significant AR lags.
        ma_lags_all : list of int
            Individually significant MA lags.
        grid_fit_ar : pd.DataFrame
            Per-model grid for the pure AR search.
        grid_fit_ma : pd.DataFrame
            Per-model grid for the pure MA search.

    For ``search_stage="barma1"`` or ``"barma2"``:
        grid_fit_ord : pd.DataFrame
            The AR-side and MA-side candidates combined, sorted ascending by BIC
            with a reset index.

    Raises
    ------
    ValueError
        If ``search_stage`` is not one of "barma0", "barma1", "barma2".

    See Also
    --------
    search_component : the per-component worker used by each stage.
    """

    if search_stage == "barma0":
        # Stage 0: fit the pure AR and pure MA models individually.
        (grid_fit_ar, ar_lags_all) = search_component(
            y=y,
            exog=exog,
            component="AR",
            start_lag=start_lag,
            end_lag=end_lag,
            sig_level=sig_level,
        )

        (grid_fit_ma, ma_lags_all) = search_component(
            y=y,
            exog=exog,
            component="MA",
            start_lag=start_lag,
            end_lag=end_lag,
            sig_level=sig_level,
        )

        return ar_lags_all, ma_lags_all, grid_fit_ar, grid_fit_ma

    elif search_stage == "barma1":
        # Stage 1: fix the best lag of one component, search the other.
        # AR side: fix the best MA lag, add each candidate AR lag.
        (grid_fit_ar, _) = search_component(
            y=y,
            exog=exog,
            component="AR",
            start_lag=start_lag,
            end_lag=end_lag,
            sig_level=sig_level,
            ar_lags_fixed=None,
            ma_lags_fixed=ma_lags_fixed,
            ar_lags_all=ar_lags_all,
            ma_lags_all=ma_lags_all,
        )
        # MA side: fix the best AR lag, add each candidate MA lag.
        (grid_fit_ma, _) = search_component(
            y=y,
            exog=exog,
            component="MA",
            start_lag=start_lag,
            end_lag=end_lag,
            sig_level=sig_level,
            ar_lags_fixed=ar_lags_fixed,
            ma_lags_fixed=None,
            ar_lags_all=ar_lags_all,
            ma_lags_all=ma_lags_all,
        )

        grid_fit = pd.concat([grid_fit_ar, grid_fit_ma])
        grid_fit = grid_fit.drop_duplicates(subset=["AIC", "BIC"])
        grid_fit_ord = grid_fit.sort_values("BIC").reset_index(drop=True)
        grid_fit_ord.index = np.arange(1, len(grid_fit) + 1, 1)

        return grid_fit_ord

    elif search_stage == "barma2":
        # Stage 2: keep both best components fixed and add one more lag to each.
        # AR side.
        (grid_fit_ar, _) = search_component(
            y=y,
            exog=exog,
            component="AR",
            start_lag=start_lag,
            end_lag=end_lag,
            sig_level=sig_level,
            ar_lags_fixed=ar_lags_fixed,
            ma_lags_fixed=ma_lags_fixed,
            ar_lags_all=ar_lags_all,
            ma_lags_all=ma_lags_all,
        )
        # MA side
        (grid_fit_ma, _) = search_component(
            y=y,
            exog=exog,
            component="MA",
            start_lag=start_lag,
            end_lag=end_lag,
            sig_level=sig_level,
            ar_lags_fixed=ar_lags_fixed,
            ma_lags_fixed=ma_lags_fixed,
            ar_lags_all=ar_lags_all,
            ma_lags_all=ma_lags_all,
        )

        grid_fit = pd.concat([grid_fit_ar, grid_fit_ma])
        grid_fit = grid_fit.drop_duplicates(subset=["AIC", "BIC"])
        grid_fit_ord = grid_fit.sort_values("BIC").reset_index(drop=True)
        grid_fit_ord.index = np.arange(1, len(grid_fit) + 1, 1)

        return grid_fit_ord

    else:
        raise ValueError(
            "The search stage available are: 'barma0', 'barma1', 'barma2'."
            f"The search stage {search_stage} is not recognized."
        )


def residuals_top_models(grid_fit, y, exog, white_noise=False):
    """
    Render a Quarto tabset with diagnostics for the top-5 BARMA models by BIC.

    For each of the 5 lowest-BIC candidate models in `grid_fit`, this function
    refits a BARMA model on the training data and displays, inside its own
    Quarto tab, the model summary and residual diagnostic plots (Pearson
    residuals): a standard diagnostics plot and a Ljung-Box p-value plot.

    Parameters
    ----------
    grid_fit : pd.DataFrame or array-like convertible to pd.DataFrame
        Grid-search results from the model selection step. Must contain at
        least the columns "AR", "MA", and "BIC", where each row represents
        one candidate (AR, MA) order combination and its associated BIC.

    Returns
    -------
    grid_fit_top : pd.DataFrame
        The 5 rows of `grid_fit` with the lowest BIC, sorted ascending and
        with a reset integer index.

    Notes
    -----
    - This function is side-effect only: it calls `display()` to render
      Markdown, model summaries, and matplotlib figures directly into the
      Quarto document. It does not return the fitted models themselves.
    - `y` and `exog` are not function parameters — they are read
      from the enclosing notebook/report scope and must be defined before
      calling this function.
    - The number of models shown (5) and the residual type used for the
      diagnostic plots ("pearson") are hardcoded, since this function is
      built for this specific report rather than for general reuse.
    - Must be run inside a Quarto/Jupyter cell: it relies on
      `IPython.display.display` and Quarto's fenced div syntax
      (`::: {.panel-tabset}`) to build the tabbed layout.

    Examples
    --------
    >>> top5 = residuals_top_models(phase2_grid_fit_ord)
    >>> top5[["AR", "MA", "BIC"]]
    """
    grid_fit = pd.DataFrame(grid_fit)

    # Keep only the 5 candidate models with the lowest BIC (best fit).
    # grid_fit_top = grid_fit.sort_values(by="BIC").head(10).reset_index(drop=True)
    # grid_fit_top = grid_fit.sort_values(by="BIC").reset_index(drop=True)
    grid_fit_top = grid_fit

    # Open a Quarto tabset: each candidate model gets its own tab below.
    display(Markdown("::: {.panel-tabset}"))

    white_noise_list = []
    for row in range(len(grid_fit_top)):
        ar_lag = grid_fit_top.iloc[row]["AR"]
        ma_lag = grid_fit_top.iloc[row]["MA"]

        # Refit the model for this (AR, MA) combination. y_train and X_train
        # come from the notebook scope, not from grid_fit.
        barma_model = BARMA(
            y=y,
            ar=ar_lag,
            ma=ma_lag,
            exog=exog,
        )

        barma_fit = barma_model.fit()
        summary = barma_fit.summary()

        # -----------------------------------------------------------------------------
        # Check white noise
        # -----------------------------------------------------------------------------

        fig_plot_diagnostics, spike_correlation = barma_fit.plot_diagnostics(
            resid_type="pearson"
        )

        if not white_noise or not spike_correlation:
            white_noise_list.append(
                {
                    "AR": ar_lag,
                    "MA": ma_lag,
                    "Conv": barma_fit.converged,
                    "AIC": barma_fit.aic,
                    "BIC": barma_fit.bic,
                }
            )

            display(Markdown((f"\n## AR({ar_lag}), MA: {ma_lag}\n")))

            # Standard residual diagnostics
            display(Markdown(("Summary")))
            display(summary)

            display(fig_plot_diagnostics)
            plt.close(fig_plot_diagnostics)

            # Ljung-Box p-values across lags
            fig_plot_ljungbox = barma_fit.plot_ljungbox(resid_type="pearson")

            display(fig_plot_ljungbox)
            plt.close(fig_plot_ljungbox)

            print("\nljungbox_test: \n")
            display(pd.DataFrame(barma_fit.ljungbox_test().round(4)))

        plt.close(fig_plot_diagnostics)

    white_noise_df = pd.DataFrame(
        white_noise_list, columns=["AR", "MA", "Conv", "AIC", "BIC"]
    )

    # display(Markdown(white_noise_df.to_markdown()))
    display(Markdown((f"\n### AR({ar_lag}), MA: {ma_lag}\n")))
    display(Markdown("\n" + white_noise_df.to_markdown() + "\n"))

    # Close the Quarto tabset.
    display(Markdown(":::"))

    return grid_fit_top

    # =================================================================================


def plot_diagnostics(y, fitted_values, residuals):

    # -----------------------------------------------------------------------------
    # 1. Extract the model estimates and link function
    # -----------------------------------------------------------------------------
    y_index = y.index

    # -----------------------------------------------------------------------------
    # 2. Plot configuration
    # -----------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))

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

    axes[0, 1].set_title("Residuals over time", fontsize=11)

    # Aux values for setting the limits of the plot
    min_tick = np.abs(np.min(residuals))
    max_tick = np.abs(np.max(residuals))
    tick_lim = np.ceil(max(min_tick, max_tick))

    axes[0, 1].set_xlabel("Time", fontsize=8)
    axes[0, 1].set_ylabel("Residuals", fontsize=8)

    axes[0, 1].tick_params(labelsize=8)
    axes[0, 1].axhline(y=0, linewidth=0.5, linestyle="-", color="black")

    axes[0, 1].set_yticks(np.arange(-tick_lim, tick_lim, step=1))

    # -----------------------------------------------------------------------------
    # Panel [1,0]: Residual ACF
    # -----------------------------------------------------------------------------
    plot_acf(residuals, lags=24, ax=axes[1, 0], zero=False, alpha=None)

    # Aux values for setting the limits of the plot
    ci_bound = 1.96 / np.sqrt(len(residuals))
    acf_values = acf(residuals, nlags=24)
    acf_values_plot = acf_values[1:]
    acf_values_plot_limits = np.max(np.abs(acf_values_plot)) + 0.05

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
        residuals,
        lags=24,
        ax=axes[1, 1],
        zero=False,
        alpha=None,
    )

    # Aux values for setting the limits of the plot
    pacf_values = pacf(residuals, nlags=24)
    pacf_values_plot = pacf_values[1:]
    pacf_values_plot_limits = np.max(np.abs(pacf_values_plot)) + 0.05

    axes[1, 1].set_xlabel("Lags", fontsize=8)
    axes[1, 1].set_ylabel("PACF", fontsize=8)

    axes[1, 1].axhline(ci_bound, color="blue", linestyle="dashed", linewidth=0.8)
    axes[1, 1].axhline(-ci_bound, color="blue", linestyle="dashed", linewidth=0.8)

    axes[1, 1].tick_params(labelsize=8)
    axes[1, 1].set_ylim(-pacf_values_plot_limits, pacf_values_plot_limits)
    axes[1, 1].set_xticks(np.arange(0, 25, step=6))

    fig.tight_layout(h_pad=2.0, w_pad=2.0)

    # return fig, acf_values_plot.round(4), pacf_values_plot.round(4), ci_bound.round(4)
    return fig


def plot_forecast(
    y_test: pd.Series, forecast: pd.Series | pd.DataFrame | np.array
) -> plt.Figure:
    """
    Plot the observed test data against the forecast values.

    Parameters:
    -----------
    y_test: pd.Series
        The observed, the true values, ideally with DateTimeIndex.
    forecast: pd.Series | pd.DataFrame | np.ndarray
        The predicted values corresponding to the y_test period.

    Returns
    -----------
    fig: plt.Figure
        The matplotlib figure object.
    """

    # -----------------------------------------------------------------------------
    # 1. Plot configuration
    # -----------------------------------------------------------------------------
    fig, axes = plt.subplots(figsize=(8, 6))

    # Create a vector for index label
    if isinstance(y_test, pd.DataFrame):
        forecast_index = y_test.index
    else:
        forecast_index = np.arange(1, len(y_test) + 1)

    forecast = pd.DataFrame(forecast, index=forecast_index)

    # -----------------------------------------------------------------------------
    # 2. Plotting
    # -----------------------------------------------------------------------------
    axes.plot(
        y_test.index,
        y_test.values,
        color="steelblue",
        linewidth=0.8,
        label="Observed",
    )
    axes.plot(
        forecast.index,
        forecast.values,
        color="salmon",
        linewidth=0.8,
        label="Forecast",
    )

    axes.set_title(" ", fontsize=11)
    axes.set_xlabel("Time", fontsize=8)
    axes.set_ylabel("Relative Humidity", fontsize=8)

    # Legend
    axes.legend(loc="best", fontsize=11)
    axes.tick_params(labelsize=8)
    fig.tight_layout()

    return fig


if __name__ == "__main__":
    # ---------------------------------------------------------------------------------
    # Data
    # ---------------------------------------------------------------------------------
    PROCESSED_DIR = Path("data/processed")

    y_full: pd.Series = pd.read_csv(
        PROCESSED_DIR / "y_full_monthly_data.csv",
        index_col="Month",
        parse_dates=["Month"],
    ).squeeze("columns")
    y_full.name = "y"
    y_full.index.freq = pd.infer_freq(y_full.index)

    # split the data set with 80% for train and 20% for test
    split_index = int(len(y_full) * 0.8)
    y_train = y_full.iloc[:split_index]
    y_test = y_full.iloc[split_index:]

    link = "logit"

    # ---------------------------------------------------------------------------------
    # Regressors (deterministic seasonal harmonics)
    # ---------------------------------------------------------------------------------
    freq = 12
    n_obs_train_sub = len(y_train)

    # Trend index starts at 1 to match the R reference.
    trend_index_train = np.arange(1, n_obs_train_sub + 1)
    hs_train: np.ndarray = np.sin(2 * np.pi * trend_index_train / freq)
    hc_train: np.ndarray = np.cos(2 * np.pi * trend_index_train / freq)

    X_train: np.ndarray = np.column_stack((hs_train, hc_train))

    print(f"The shape of the X_train matrix is: {X_train.shape}")

    n_test = len(y_test)

    index_test_start = np.max(trend_index_train) + 1
    index_end_start = np.max(trend_index_train) + n_test + 1
    trend_index_test = np.arange(index_test_start, index_end_start, 1)

    hs_test: np.ndarray = np.sin(2 * np.pi * trend_index_test / freq)
    hc_test: np.ndarray = np.cos(2 * np.pi * trend_index_test / freq)

    X_test_array: np.ndarray = np.column_stack((hs_test, hc_test))

    X_test: pd.DataFrame = pd.DataFrame(
        X_test_array, index=y_test.index, columns=["hs", "hc"]
    )

    print(f"The shape of the X_test matrix is: {X_test.shape}")

    # ---------------------------------------------------------------------------------
    # Phase 0
    # ---------------------------------------------------------------------------------
    start_lag = 1
    end_lag = 12
    sig_level = 1.00

    # Step 0: Fit the AR and MA individuals
    (
        phase0_ar_lags_all,
        phase0_ma_lags_all,
        phase0_grid_fit_ar,
        phase0_grid_fit_ma,
    ) = auto_barma(
        y=y_train,
        exog=X_train,
        start_lag=start_lag,
        end_lag=end_lag,
        sig_level=sig_level,
        search_stage="barma0",
    )

    # ---------------------------------------------------------------------------------
    # Phase 1: Fit the AR and MA individuals
    # ---------------------------------------------------------------------------------
    phase0_ar_lags_fixed = phase0_grid_fit_ar.sort_values("BIC").iloc[0]["AR"]
    phase0_ma_lags_fixed = phase0_grid_fit_ma.sort_values("BIC").iloc[0]["MA"]

    phase1_grid_fit_ord = auto_barma(
        y=y_train,
        exog=X_train,
        start_lag=start_lag,
        end_lag=end_lag,
        sig_level=sig_level,
        ar_lags_fixed=phase0_ar_lags_fixed,
        ma_lags_fixed=phase0_ma_lags_fixed,
        ar_lags_all=phase0_ar_lags_all,
        ma_lags_all=phase0_ma_lags_all,
        search_stage="barma1",
    )

    # ---------------------------------------------------------------------------------
    # Phase 2: Fit the AR and MA individuals
    # ---------------------------------------------------------------------------------

    phase1_ar_lags_fixed = phase1_grid_fit_ord.sort_values("BIC").iloc[0]["AR"]
    phase1_ma_lags_fixed = phase1_grid_fit_ord.sort_values("BIC").iloc[0]["MA"]

    phase2_grid_fit_ord = auto_barma(
        y=y_train,
        exog=X_train,
        start_lag=start_lag,
        end_lag=end_lag,
        sig_level=sig_level,
        ar_lags_fixed=phase1_ar_lags_fixed,
        ma_lags_fixed=phase1_ma_lags_fixed,
        ar_lags_all=phase0_ar_lags_all,
        ma_lags_all=phase0_ma_lags_all,
        search_stage="barma2",
    )

    # ---------------------------------------------------------------------------------
    # Fit the DHR model
    # ---------------------------------------------------------------------------------
    #| label: dhr_fit
    #| include: true
    #| echo: true

    # Approach 1: sarima
    dhr_fit = auto_arima(
        y=y_train,
        X=X_train,
        d=0,
        D=0,
        with_intercept=True,
        trend=None,
        seasonal=None,
        m=12,
    )

    display(dhr_fit.summary())

    # Compute the MAE and RMSE of SARIMA model fit.
    #| label: y_hat_fit_dhr

    y_hat_dhr_fit = dhr_fit.predict(n_periods=len(y_test), X=X_test)

    rmse_dhr = np.sqrt(np.mean((y_test.values - y_hat_dhr_fit.values) ** 2))
    mae_dhr = np.mean(np.abs(y_test.values - y_hat_dhr_fit.values))

    print(f"RMSE_dhr: {rmse_dhr.round(4)}")
    print(f"MAE_dhr:  {mae_dhr.round(4)}")

    # Residual analysis of DHR.

    #| label: plot_fit_dhr

    plot_dhr_fit = plot_diagnostics(
        y=y_train,
        fitted_values=y_hat_dhr_fit,
        residuals=dhr_fit.resid(),
    )

    display(plot_dhr_fit)
    plt.close(plot_dhr_fit)

    # Compute the MAE and RMSE of ARIMA fit.

    # ---------------------------------------------------------------------------------
    # SARIMA
    # ---------------------------------------------------------------------------------
    # Fit the SARIMA model

    # Approach 2: ARIMA
    sarima_fit = auto_arima(
        y=y_train, d=0, D=0, with_intercept=True, trend=None, seasonal=True, m=12
    )

    display(sarima_fit.summary())

    y_hat_sarima_fit = sarima_fit.predict(n_periods=len(y_test))

    rmse_sarima = np.sqrt(np.mean((y_test.values - y_hat_sarima_fit.values) ** 2))
    mae_sarima = np.mean(np.abs(y_test.values - y_hat_sarima_fit.values))

    # Residual analysis of ARIMA fit.
    plot_fit_sarima = plot_diagnostics(
        y=y_train, fitted_values=y_hat_sarima_fit, residuals=sarima_fit.resid()
    )

    display(plot_fit_sarima)
    plt.close(plot_fit_sarima)

    # ---------------------------------------------------------------------------------
    # BARMA
    # ---------------------------------------------------------------------------------
    ar_lag = [1, 21, 24]
    ma_lag = [1, 26]

    barma_model = BARMA(
        y=y_train,
        ar=ar_lag,
        ma=ma_lag,
        exog=X_train,
    )

    barma_fit = barma_model.fit()
    summary = barma_fit.summary()

    display(summary)

    # Compute the MAE and RMSE
    forecast_barma = barma_fit.forecast(h=len(y_test.values), exog=X_test)
    rmse_barma = np.sqrt(np.mean((y_test.values - forecast_barma.values) ** 2))
    mae_barma = np.mean(np.abs(y_test.values - forecast_barma.values))

    # ---------------------------------------------------------------------------------
    ### Forecast comparisons
    # ---------------------------------------------------------------------------------
    #| include: true
    #| echo: false
    forecast_resume_dic = {
        "Model": ["BARMA", "SARIMAX", "ARIMA"],
        "MAE": [mae_barma, mae_dhr, mae_sarima],
        "RMSE": [rmse_barma, rmse_dhr, rmse_sarima],
    }

    forecast_resume_df = pd.DataFrame(forecast_resume_dic).round(4)

    from IPython.display import Markdown

    print(
        forecast_resume_df.to_markdown(index=False, colalign=("left", "right", "right"))
    )


y_hat_fit_barma = barma_fit.forecast(h=len(y_test.values), exog=X_test)
y_hat_sarima_fit = sarima_fit.predict(n_periods=len(y_test))

print("y_test: ", y_test.values)
print("y_hat_fit_barma: ", y_hat_fit_barma.values)
print("y_hat_dhr_fit: ", y_hat_dhr_fit.values)
print("y_hat_sarima_fit: ", y_hat_sarima_fit.values)


rmse_barma = np.sqrt(np.mean((y_test.values - y_hat_fit_barma.values) ** 2))
mae_barma = np.mean(np.abs(y_test.values - y_hat_fit_barma.values))

forecast_resume_dic = {
    "Model": ["BARMA", "DHR", "SARIMA"],
    "MAE": [mae_barma, mae_dhr, mae_sarima],
    "RMSE (%)": [rmse_barma, rmse_dhr, rmse_sarima],
}

forecast_resume_df = pd.DataFrame(forecast_resume_dic).round(4)

forecast_resume_df["RMSE (%)"] = forecast_resume_df["RMSE (%)"] * 100

forecast_resume_df.index = range(1, len(forecast_resume_df) + 1)
display(forecast_resume_df)

import seaborn as sns

sns.set_theme(
    style="whitegrid",
    context="talk",
    rc={
        "axes.titlesize": 14,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    },
)
