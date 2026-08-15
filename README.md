# BARMA-Python

<!--[![CI](https://github.com/Everton-da-Costa/BARMA-Python-2026/actions/workflows/ci.yaml/badge.svg)](https://github.com/Everton-da-Costa/BARMA-Python-2026/actions)-->
[![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

A Python implementation of the **Beta Autoregressive Moving Average** ($\beta$ARMA) model for time series bounded in the unit interval $(0, 1)$ —
rates, proportions, and relative indices. This is a Python port of the
methodology available in R via the [`betaARMA`](https://github.com/Everton-da-Costa/betaARMA)
package on CRAN, validated against the R reference to machine precision (`1e-10`).

**Headline result:** on a 66-month hold-out of monthly relative humidity in
Brasília, $\beta$ARMA achieved the lowest error of four models (MAE **8.73%**,
RMSE **10.39%**) — and is the only one whose forecasts are *guaranteed* to stay
within $(0, 1)$. See [Worked Example & Reports](#-worked-example--reports).

---

## 📚 Table of Contents

- [🎯 Project Motivation](#-project-motivation)
- [✨ Key Features](#-key-features)
- [🧭 Public API](#-public-api)
- [🛠️ Installation](#️-installation)
- [🚀 Getting Started](#-getting-started)
- [📊 Worked Example & Reports](#-worked-example--reports)
- [🧠 Key Skills Demonstrated](#-key-skills-demonstrated)
- [📂 Repository Structure](#-repository-structure)
- [🗺️ Roadmap](#️-roadmap)
- [🔗 Related Projects](#-related-projects)
- [📖 Foundational Literature](#-foundational-literature)
- [🎓 Citation](#-citation)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)
- [📬 Contact](#-contact)

---

## 🎯 Project Motivation

Time series bounded in $(0, 1)$ — proportions, rates, relative humidity,
reservoir levels — are poorly served by standard Gaussian methods like ARIMA,
which do not respect the natural boundaries of the data and can produce fitted
values or forecasts outside the admissible range. The $\beta$ARMA model
addresses this by assuming a conditional Beta distribution, guaranteeing that
fitted and predicted values stay strictly within $(0, 1)$.

This project ports the core $\beta$ARMA estimation routines from R to Python,
with two goals:

1. Make $\beta$ARMA accessible to the broader Python statistical and machine
   learning ecosystem.
2. Enable future comparisons of computational performance, numerical stability,
   and convergence behavior between the R and Python implementations.

The port reproduces the R package's core estimation routines and validates them
against the R reference via a `pytest` suite.

---

## ✨ Key Features

- **Unified $\beta$ARMA fitting**: a single `BARMA` class handles any
  combination of AR and MA lags (including non-contiguous, subset-ARMA
  specifications), pure $\beta$AR / $\beta$MA submodels, and exogenous
  regressors.
- **Analytic estimation**: conditional maximum likelihood via SciPy's BFGS
  optimizer, using the analytic log-likelihood and score vector (gradients)
  rather than numerical differentiation. The `logit` link is currently
  supported.
- **Full inference**: a Fisher Information Matrix implementation provides
  standard errors, z-values, and p-values, with AIC and BIC for model selection.
- **Diagnostics**: Pearson, raw, and scale residuals; a four-panel diagnostic
  grid (observed vs. fitted, residuals over time, ACF, PACF) and a Ljung–Box
  portmanteau test with plot.
- **Bounded forecasting**: out-of-sample forecasts that, by construction, stay
  strictly within $(0, 1)$, a guarantee ARIMA-family models do not provide.
- **Validated against R**: a `pytest` suite compares the log-likelihood, score
  vector, starting values, and link structure against the R reference at a
  `1e-10` tolerance.

---

## 🧭 Public API

Modeling is organized around five actions on the `BARMA` / `BARMAResults`
classes (both exposed at the package top level):

| Action    | Interface |
|-----------|-----------|
| Construct | `BARMA(y, ar, ma, exog, link)` |
| Fit       | `.fit()` → `BARMAResults` (with `.converged`, `.n_iter`) |
| Inspect   | `.summary()`, `.aic`, `.bic`, `.log_likelihood`, `.fitted_values`, `.fim_barma` |
| Diagnose  | `.residuals()`, `.plot_diagnostics()`, `.ljungbox_test()`, `.plot_ljungbox()` |
| Forecast  | `.forecast()`, `.plot_forecast()` |

---

## 🛠️ Installation

**Requirements:** Python 3.12+ (developed and tested on Ubuntu 24.04). Core
dependencies are NumPy, SciPy, pandas, matplotlib, and statsmodels; the forecast
benchmark additionally uses `pmdarima`.

Clone the repository and run from the project directory:

```bash
# 1. Clone the repository
git clone https://github.com/Everton-da-Costa/BARMA-Python-2026.git
cd BARMA-Python-2026

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# 3. Install the package and its dependencies (editable mode)
pip install -e ".[dev,benchmark]"

# 4. Verify the R-reference validation
pytest
```

Editable mode (`-e`) installs the package in place, so the code and the
companion reports run directly from the repository. Packaging for distribution
(PyPI) is on the [roadmap](#️-roadmap).

---

## 🚀 Getting Started

```python
import numpy as np
import pandas as pd
from src.model import BARMA

# Fit a beta-ARMA model with harmonic (seasonal) regressors
model = BARMA(
    y=y_train,            # pd.Series bounded in (0, 1)
    ar=[1, 21, 24],       # AR lags
    ma=[1, 26],           # MA lags
    exog=X_train,         # optional exogenous regressors (e.g. sine/cosine)
    link="logit",
)

fit = model.fit()

# Inspect the fitted model
print(f"Converged: {fit.converged}")
fit.summary()             # coefficients, standard errors, z-values, p-values
print(fit.aic, fit.bic)

# Residual diagnostics
fit.plot_diagnostics(resid_type="pearson")
fit.plot_ljungbox(resid_type="pearson")

# Out-of-sample forecast (stays within (0, 1) by construction)
forecast = fit.forecast(h=len(y_test), exog=X_test)
fit.plot_forecast(y_test=y_test, exog=X_test)
```

The full seasonal specification above — AR lags 1, 21, 24 and MA lags 1, 26 on
real NASA POWER data — is walked through end to end in the modeling report below.

---

## 📊 Worked Example & Reports

Two companion reports (in `report/`) demonstrate the package on real data
(monthly relative humidity in Brasília, fetched from the NASA POWER API):

- **[Modeling report](https://htmlpreview.github.io/?https://github.com/Everton-da-Costa/BARMA-Python-2026/blob/main/report/report_brasilia_relative_humidity.html)**
  — a full walkthrough: data, seasonality, model selection, fitting, residual
  diagnostics, and a forecast benchmark against DHR, SARIMA, and ARIMA baselines.
- **[Roadmap & validation report](https://htmlpreview.github.io/?https://github.com/Everton-da-Costa/BARMA-Python-2026/blob/main/report/report_roadmap.html)**
  — validation methodology (R-reference testing to machine precision) and
  project status.

> These reports render in the browser via `htmlpreview.github.io`.

### Forecast benchmark

On the 66-month test set (roughly 5.5 years), forecast accuracy was:

| Model      | MAE (%) | RMSE (%) |
|------------|:-------:|:--------:|
| BARMA      | 8.73    | 10.39    |
| DHR        | 8.90    | 10.57    |
| SARIMA     | 11.22   | 13.52    |
| ARIMA      | 11.88   | 14.38    |

$\beta$ARMA achieves the lowest error on both metrics. 
While DHR yields highly competitive point accuracy, unlike all three baselines,
$\beta$ARMA forecasts are guaranteed to remain within $(0, 1)$ by construction.
It is notable that SARIMA and ARIMA collapse toward the training mean over the horizon.
This is visualized in Figure 6 of the **[Modeling report](https://htmlpreview.github.io/?https://github.com/Everton-da-Costa/BARMA-Python-2026/blob/main/report/report_brasilia_relative_humidity.html)**.

---

## 🧠 Key Skills Demonstrated

- **Statistical software engineering**: porting a published statistical method
  from R to Python with a clean, class-based API (model/results separation) and
  a modular estimation engine (log-likelihood, score vector, starting values,
  and link structure as separate components).
- **Numerical validation**: a `pytest` suite verifying the Python
  implementation against an R reference (log-likelihood, score vector, starting
  values, link structure) at a `1e-10` tolerance.
- **Statistical inference**: translated the Fisher Information Matrix, score
  vector, and analytic log-likelihood from a validated R implementation to
  Python.
- **Time series analysis**: subset-ARMA specification, model selection by BIC,
  residual diagnostics (ACF, PACF, Ljung–Box), and out-of-sample forecasting.
- **Reproducible research**: Quarto reports as narrated, end-to-end case
  studies.

---

## 📂 Repository Structure

```plaintext
.
├── src/                        # Estimation engine and public API
│   ├── model.py                #   BARMA + BARMAResults classes
│   ├── barma.py                #   core fitting routines
│   ├── loglik_barma.py         #   analytic log-likelihood
│   ├── score_vector_barma.py   #   analytic score vector (gradients)
│   ├── make_link_structure.py  #   link function (logit)
│   ├── start_values.py         #   starting-value estimation
│   ├── fitted_barma.py         #   fitted-value computation
│   ├── config.py, utils.py     #   configuration and helpers
│   └── __init__.py             #   exposes BARMA, BARMAResults
├── tests/                      # pytest suite validating against the R reference
├── original_R_code/            # R reference implementation (validation oracle)
├── data/
│   ├── processed/              # Processed time series (.csv)
│   └── raw/                    # Raw data + R reference values (loglik, score, starts)
├── scripts/                    # fetch_humidity_brasilia.py (NASA POWER ingestion)
├── report/                     # Quarto reports (modeling walkthrough, roadmap)
├── pyproject.toml              # Package metadata and dependencies
├── Makefile                    # Common tasks (test, render reports)
├── _quarto.yml                 # Quarto project configuration
├── .env.example                # QUARTO_PYTHON template for rendering
├── LICENSE                     # MIT License
└── README.md                   # This file
```

---

## 🗺️ Roadmap

- **Package for distribution (PyPI):** expose a proper top-level package import
  (replacing the current `from src.model import BARMA`), with documentation and
  install instructions.
- **Additional link functions:** extend beyond `logit` to `cloglog`, `loglog`,
  and `probit`, matching the R reference implementation.
- **Extend benchmarking:** add machine-learning baselines — gradient-boosted
  trees (XGBoost/LightGBM) on lagged and calendar features, and a neural
  forecaster (e.g. N-BEATS via `neuralforecast`).
- **Performance & stability study:** compare R vs. Python convergence behavior
  on flat-likelihood cases — the open question that motivated this port.

---

## 🔗 Related Projects

This project is part of a broader body of work on beta autoregressive models:

- **[betaARMA](https://github.com/Everton-da-Costa/betaARMA)** — the R package on
  CRAN that this project ports; the reference implementation.
- **[BARMAJournalHydrology2024](https://github.com/Everton-da-Costa/BARMAJournalHydrology2024)** —
  R package and analysis for the *Journal of Hydrology* (2024) paper on link
  function selection and hypothesis tests.
- **[BarmaRidgeBJPS2025](https://github.com/Everton-da-Costa/BarmaRidgeBJPS2025)** —
  R package for the *Brazilian Journal of Probability and Statistics* (2025)
  paper on ridge-penalized estimation for numerical stability.

---

## 📖 Foundational Literature

- **Rocha, A. V., & Cribari-Neto, F. (2009).** "Beta autoregressive moving
  average models." *TEST*, 18(3), 529–545.
  [doi:10.1007/s11749-008-0112-z](https://doi.org/10.1007/s11749-008-0112-z)
- **Rocha, A. V., & Cribari-Neto, F. (2017).** "Erratum to: Beta autoregressive
  moving average models." *TEST*, 26(2), 451–459.
  [doi:10.1007/s11749-017-0528-4](https://doi.org/10.1007/s11749-017-0528-4)
- **Costa, E., Cribari-Neto, F., & Scher, V. T. (2024).** "Test inferences and
  link function selection in dynamic beta modeling of seasonal
  hydro-environmental time series with temporary abnormal regimes." *Journal of
  Hydrology*, 638, 131489.
  [doi:10.1016/j.jhydrol.2024.131489](https://doi.org/10.1016/j.jhydrol.2024.131489)
- **Cribari-Neto, F., Costa, E., & Fonseca, R. V. (2025).** "Numerical stability
  enhancements in beta autoregressive moving average model estimation."
  *Brazilian Journal of Probability and Statistics*, 39(4), 410–437.
  [doi:10.1214/25-BJPS645](https://doi.org/10.1214/25-BJPS645)

---

## 🎓 Citation

If you use this software in your research, please cite the repository:

```bibtex
@software{Costa_BARMA_Python_2026,
  title   = {{BARMA-Python}: A Python Implementation of Beta Autoregressive
             Moving Average Models},
  author  = {Costa, E.},
  year    = {2026},
  version = {1.0.0},
  url     = {https://github.com/Everton-da-Costa/BARMA-Python-2026}
}
```

This project is a direct Python port of the `betaARMA` reference implementation on CRAN.
If your work utilizes or compares against the R ecosystem, please consider citing the 
original package:

```bibtex
@Manual{Costa_Cribari_Scher2026,
  title   = {betaARMA: Beta Autoregressive Moving Average Models},
  author  = {Costa, E. and Cribari-Neto, F. and Scher, V. T.},
  year    = {2026},
  note    = {R package version 1.2.0},
  doi     = {10.32614/CRAN.package.betaARMA},
  url     = {[https://CRAN.R-project.org/package=betaARMA](https://CRAN.R-project.org/package=betaARMA)}
}
```

The implementation is based on the methodology introduced by
Rocha & Cribari-Neto (2009); if you reference the model, please also cite:

```bibtex
@Article{Rocha_Cribari_2009,
  title   = {Beta autoregressive moving average models},
  author  = {Rocha, A. V. and Cribari-Neto, F.},
  journal = {TEST},
  year    = {2009},
  volume  = {18},
  number  = {3},
  pages   = {529--545},
  doi     = {10.1007/s11749-008-0112-z}
}
```

---

## 🤝 Contributing

Contributions are welcome. If you find an issue or have a suggestion, please open
an issue or submit a pull request.

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file
for details.

## 📬 Contact

**Everton da Costa**
- 💼 [LinkedIn](https://linkedin.com/in/everton-da-costa)
- 📧 <everto.cost@gmail.com>