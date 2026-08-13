#' @title Score Vector for a BARMA Model
#' @description This function computes the score vector for a Beta
#'   Autoregressive Moving Average (BARMA) model. The score vector is the
#'   gradient of the log-likelihood function with respect to the model
#'   parameters.
#'
#' @param y A time series of data, with values in the interval (0, 1).
#' @param ar A numeric vector of positive integers specifying the AR lags.
#' @param ma A numeric vector of positive integers specifying the MA lags.
#' @param alpha The numeric intercept term in the linear predictor.
#' @param varphi A numeric vector of AR parameters.
#' @param theta A numeric vector of MA parameters.
#' @param phi The numeric positive precision parameter.
#' @param link A character string for the link function, e.g., "logit".
#'
#' @return A numeric vector containing the score values, ordered as
#'   (alpha, varphi, theta, phi).
#' @keywords internal
score_vector_barma <- function(y, ar, ma,
                              alpha = 0, varphi = 0, theta = 0,
                              phi = 0, link) {

  # Link functions setup
  # ----------------------------------------------------------------------- #
  link_structure <- make_link_structure(link)
  linkfun <- link_structure$linkfun
  linkinv <- link_structure$linkinv
  mu.eta <- link_structure$mu.eta

  ynew <- linkfun(y)

  # Model dimensions
  # ------------------------------------------------------------------------- #
  p <-  max(ar)
  q <-  max(ma)
  p1 <- length(ar)
  q1 <- length(ma)
  n <- length(y)
  m <- max(p, q, na.rm = TRUE)

  # Initialize vectors for the recursive calculations
  # ------------------------------------------------------------------------- #
  error <- rep(0, n)
  eta <- rep(NA, n)

  # Recursively compute the linear predictor (eta) and errors
  for (t in (m + 1):n) {
    eta[t] <- alpha +
      crossprod(varphi, ynew[t - ar]) +
      crossprod(theta, error[t - ma])
    error[t]  <- ynew[t] - eta[t]
  }

  # Subset all series to the effective sample size (n - m)
  eta1 <- eta[(m + 1):n]
  mu1  <- linkinv(eta = eta1)
  y1   <- y[(m + 1):n]

  # Derivatives of the log-likelihood function
  # ------------------------------------------------------------------------- #
  # Design matrices for AR (P) and MA (R) components
  P <- matrix(nrow = n - m, ncol = p1)
  for (t in 1:(n - m)) P[t, ] <- ynew[t + m - ar]

  R <- matrix(nrow = n - m, ncol = q1)
  for (t in 1:(n - m)) R[t, ] <- error[t + m - ma]

  # Recursively compute the derivatives of eta w.r.t. each parameter
  deta_dalpha  <- rep(0, n)
  deta_dvarphi <- matrix(0, nrow = n, ncol = p1)
  deta_dtheta  <- matrix(0, nrow = n, ncol = q1)

  for (t in (m + 1):n) {
    deta_dalpha[t]    <- 1 - crossprod(theta, deta_dalpha[t - ma])
    deta_dvarphi[t, ] <- P[t - m, ] - crossprod(theta, deta_dvarphi[t - ma, ])
    deta_dtheta[t, ]  <- R[t - m, ] - crossprod(theta, deta_dtheta[t - ma, ])
  }

  # Subset derivatives to the effective sample size
  s  <- deta_dalpha[(m + 1):n]
  rP <- deta_dvarphi[(m + 1):n, ]
  rR <- deta_dtheta[(m + 1):n, ]

  # Compute score vector components
  # ------------------------------------------------------------------------- #
  mu_eta <- mu.eta(eta = eta1)
  ystar  <- log(y1 / (1 - y1))
  mustar <- digamma(mu1 * phi) - digamma((1 - mu1) * phi)

  ystar_mustar <- ystar - mustar
  mT_ystar_mustar <- mu_eta * ystar_mustar

  # Score component for each parameter
  U_alpha  <- phi * crossprod(s, mT_ystar_mustar)
  U_varphi <- phi * crossprod(rP, mT_ystar_mustar)
  U_theta  <- phi * crossprod(rR, mT_ystar_mustar)
  U_phi   <- sum(
    mu1 * ystar_mustar + log(1 - y1) -
      digamma((1 - mu1) * phi) +
      digamma(phi)
  )

  # Combine into a single score vector
  escore_vec <- c(U_alpha, U_varphi, U_theta, U_phi)

  return(escore_vec)
}
