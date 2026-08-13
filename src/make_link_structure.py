"""Link function utilities for generalized linear models.

This module provides factory functions that return the triplet of
functions required by BARMA estimation routines: the link function,
its inverse, and the derivative of the inverse with respect to the
linear predictor.

Currently supported link functions:
    - logit: Maps (0, 1) to the real line via log(mu / (1 - mu)).
"""

from typing import Callable
from scipy.special import expit
import numpy as np

__all__ = ["make_link_structure"]


def make_link_structure(link: str) -> tuple[Callable, Callable, Callable]:
    """Create a triplet of link functions for use in BARMA models.

    Returns three callables that together define a link function and its
    derivatives, as required by the BARMA log-likelihood and score vector
    routines. Currently supports the logit link.

    Parameters
    ----------
    link : str
        Name of the link function. Supported values:

        - ``'logit'``: Logistic link for responses bounded in (0, 1).

    Returns
    -------
    linkfun : Callable
        Maps the mean ``mu`` in (0, 1) to the linear predictor scale
        ``eta``. For logit: ``eta = log(mu / (1 - mu))``.
    linkinv : Callable
        Inverse link. Maps ``eta`` on the real line back to the mean
        scale ``mu`` in (0, 1).
        For logit: ``mu = exp(eta) / (1 + exp(eta))``.
    mu_eta : Callable
        Derivative of ``linkinv`` with respect to ``eta``,
        i.e. ``d(mu)/d(eta)``.
        For logit: ``exp(eta) / (1 + exp(eta))^2``.

    Raises
    ------
    ValueError
        If ``link`` is not a supported link function name.
    ValueError
        If ``mu`` values passed to the returned ``linkfun`` are not
        strictly in (0, 1).

    Examples
    --------
    >>> linkfun, linkinv, mu_eta = make_link_structure('logit')
    >>> float(linkinv(0.0))
    0.5
    >>> float(linkfun(0.5))
    0.0
    """

    if link == "logit":

        def linkfun(mu: np.ndarray) -> np.ndarray:
            # Logit link: maps mu in (0, 1) to the real line.
            if np.any((mu <= 0) | (mu >= 1)):
                raise ValueError(
                    f"mu must be strictly in (0, 1); got value(s) outside range. "
                    f"min={np.min(mu):.6g}, max={np.max(mu):.6g}"
                )
            return np.log(mu / (1 - mu))

        def linkinv(eta: np.ndarray) -> np.ndarray:
            return expit(eta)

        def mu_eta(eta: np.ndarray) -> np.ndarray:
            s = expit(eta)
            return s * (1 - s)

        return linkfun, linkinv, mu_eta

    else:
        raise ValueError(f"Link function '{link}' is not supported.")
