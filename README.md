# BARMA-Python

<!-- Badges: enable once the repo is public and CI is configured.
[![CI](https://github.com/Everton-da-Costa/BARMA-Python-2026/actions/workflows/ci.yaml/badge.svg)](https://github.com/Everton-da-Costa/BARMA-Python-2026/actions)
-->
[![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

A Python implementation of the **Beta Autoregressive Moving Average
($\beta$ARMA)** model for time series bounded in the unit interval $(0, 1)$ —
rates, proportions, and relative indices. This is a Python port of the
methodology available in R via the [`betaARMA`](https://github.com/Everton-da-Costa/betaARMA)
package on CRAN, validated against the R reference to machine precision.

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

The port reproduces the R package's core estimation routines and **validates
them against the R reference to machine precision** via a `pytest` suite.

---

## ✨ Key Features

- **Unified $\beta$ARMA fitting**: a single `BARMA` class handles any
  combination of AR and MA lags, including exogenous regressors.
- **Full inference**: a Fisher Information Matrix implementation provides
  standard errors, z-values, and p-values, with AIC and BIC for model selection.
- **Diagnostics**: Pearson, raw, and link-scale residuals; a four-panel
  diagnostic grid (observed vs. fitted, residuals over time, ACF, PACF) and a
  Ljung–Box portmanteau test with plot.
- **Bounded forecasting**: out-of-sample forecasts that, by construction, stay
  strictly within $(0, 1)$, a guarantee ARIMA-family models do not provide.
- **Validated against R**: a `pytest` suite compares the log-likelihood, score
  vector, starting values, and link structure against the R reference at a
  `1e-10` tolerance.

---

## 🧭 Public API

The user-facing interface is organized around five actions:

| Action    | Interface |
|-----------|-----------|
| Construct | `BARMA(y, ar, ma, exog, link)` |
| Fit       | `.fit()` |
| Inspect   | `.summary()`, `.aic`, `.bic`, `.log_likelihood`, `.fitted_values`, `.fim_barma` |
| Diagnose  | `.residuals()`, `.plot_diagnostics()`, `.ljungbox_test()`, `.plot_ljungbox()` |
| Forecast  | `.forecast()`, `.plot_forecast()` |

---

## 🛠️ Installation

**Requirements:** Python 3.12+ (developed and tested on Ubuntu 24.04).

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
(PyPI) is on the roadmap.

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

---

## 📊 Worked Example & Reports

Two companion reports (in `report/`) demonstrate the package on real data
(monthly relative humidity in Brasília, fetched from the NASA POWER API):

- **[Modeling report](report/report_brasilia_relative_humidity.html)** — a full
  walkthrough: data, seasonality, model selection, fitting, residual
  diagnostics, and a forecast benchmark against DHR, SARIMA, and ARIMA baselines.
- **[Roadmap](report/report_roadmap.html)** — validation methodology
  (R-reference testing to machine precision) and project status.

*(For browser viewing without cloning, render these via GitHub Pages or
`htmlpreview.github.io`, as in the related R projects.)*

---

## 🧠 Key Skills Demonstrated

- **Statistical software engineering**: porting a published statistical method
  from R to Python with a clean, class-based API (model/results separation).
- **Numerical validation**: a `pytest` suite verifying the Python
  implementation against an R reference at machine precision (`1e-10`).
- **Statistical inference**: translated the Fisher Information Matrix, score
  vector, and analytic log-likelihood from a validated R implementation to
  Python, verified against the R reference.
- **Time series analysis**: subset-ARMA specification, model selection via BIC,
  residual diagnostics (ACF, PACF, Ljung–Box), and out-of-sample forecasting.
- **Reproducible research**: Quarto reports as narrated, end-to-end case
  studies.

---

## 📂 Repository Structure

```plaintext
.
├── src/                # Source code (model.py: BARMA and results classes).
├── tests/              # pytest suite validating against the R reference.
├── original_R_code/    # R reference implementation used for validation.
├── data/
│   ├── processed/      # Processed time series (.csv).
│   └── raw/            # Raw data and reference values from R.
├── scripts/            # Data-fetching scripts (fetch_humidity_brasilia.py).
├── report/             # Quarto reports (modeling walkthrough, roadmap).
├── pyproject.toml      # Package metadata and dependencies.
├── Makefile            # Common tasks (test, render reports).
├── _quarto.yml         # Quarto project configuration.
├── LICENSE             # MIT License.
└── README.md           # This file.
```

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

If you use this code in your research, please cite the underlying methodology:

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
📧 <everto.cost@gmail.com>
