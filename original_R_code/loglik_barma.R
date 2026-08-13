#' @param y Data, a time series of numbers in (0,1).
#' @param ar A vector specifying the autoregressive (AR) component.

#' @title Log-Likelihood for a BARMA Model
#' @description This function computes the log-likelihood of the BARMA model.
#' @param ma A vector specifying the moving average (MA) component.
#' @param alpha The intercept term.
#' @param varphi A vector of autoregressive (AR) parameters.
#' @param theta A vector of moving average (MA) parameters.
#' @param phi The precision parameter of the BARMA model.
#' @param link The link function ("logit", "probit", "loglog", "cloglog").
#'
#' @importFrom stats dbeta
#'
#' @return The likelihood of the BARMA estimators.
#' @keywords internal
loglik_barma <- function(y, ar, ma,
                        alpha = 0, varphi = 0, theta = 0,
                        phi = 0, link) {

  # Link functions
  # ----------------------------------------------------------------------- #
  link_structure <- make_link_structure(link)

  linkfun <- link_structure$linkfun
  linkinv <- link_structure$linkinv
  mu.eta  <- link_structure$mu.eta

  ynew <- linkfun(y)

  # ----------------------------------------------------------------------- #
  p <-  max(ar)
  q <-  max(ma)

  n <- length(y)
  m <- max(p, q, na.rm = TRUE)

  # --------------------------------------------------------------------- #
  error <- rep(0, n)
  eta   <- rep(NA, n)
  mu    <- rep(NA, n)

  for (t in (m + 1):n) {

    eta[t] <- (alpha +
                 crossprod(varphi, ynew[t - ar]) +
                 crossprod(theta, error[t - ma]))

    error[t]  <- ynew[t] - eta[t]
  }

  mu1  <- linkinv(eta = eta[(m + 1):n])
  y1   <- y[(m + 1):n]

  ll_terms_arma <- dbeta(y1, mu1 * phi, (1 - mu1) * phi, log = TRUE)

  final <- sum(ll_terms_arma)

  return(final)
}

