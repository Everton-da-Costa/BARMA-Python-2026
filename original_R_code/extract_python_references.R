# ===========================================================================
# Oracle extraction for the Python betaARMA port
# ---------------------------------------------------------------------------
# Produces the reference (oracle) values that the Python test suite will
# assert against:
#   - the 0.8 training slice of the Brasilia relative-humidity series
#   - the start values from start_values() at that slice
#   - the log-likelihood at those start values   (16-digit reference)
#   - the score vector at those start values     (16-digit reference)
#
# All four are written to CSV so the pytest fixtures load a single source of
# truth instead of hard-coding magic numbers.
#
# Model: the reduced specification from the humidity_brasilia vignette,
#   AR = c(10, 18), MA = c(1, 13), with two harmonic regressors.
#
# Convention note (read before trusting the numbers):
#   The harmonic regressors use a trend index that RESTARTS AT 1 at the
#   beginning of the training slice, matching the vignette's
#   `seq_along(y_train)` construction. If the Python side anchors the
#   harmonics to the calendar / full-series origin instead, the beta start
#   values (and hence the log-lik and score) will differ. Keep the two sides
#   on the same convention.
# ===========================================================================

# --------------------------------------------------------------------------
# 1. Load the Brasilia series and the functions
# --------------------------------------------------------------------------
# The vignette uses `brasilia_ts` (a package dataset). 
# It lives in the betaARMA package:
library(betaARMA)

loglik_barma <- betaARMA::loglik_barma
score_vector_barma <- betaARMA::score_vector_barma
start_values <- betaARMA::start_values
make_link_structure <- betaARMA::make_link_structure

data("brasilia_ts", package = "betaARMA")
data("brasilia_df", package = "betaARMA")

# Output directory for the oracle CSVs
out_dir <- "../data/raw"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)


# --------------------------------------------------------------------------
# 2. Clean 80% split of the FULL series
# --------------------------------------------------------------------------
y_subsample_ts <- window(
  brasilia_ts,
  start = c(2006, 02),
  end   = c(2021, 09)
)

split_index = round(length(y_subsample_ts) * 0.8, 0)

y_train <- y_subsample_ts[1:split_index]
y_test <- y_subsample_ts[split_index:length(y_subsample_ts)]

y_train <- ts(y_train, start = c(2006, 02), frequency = 12)

length(y_train) / length(brasilia_ts)

cat("Full series length :", length(y_subsample_ts), "\n")
cat("Train length (0.8) :", length(y_train), "\n")

# --------------------------------------------------------------------------
# 3. Harmonic regressors (restart-at-1 convention, as in the vignette)
# --------------------------------------------------------------------------
freq        <- frequency(brasilia_ts)          # 12 for monthly
trend_index <- seq_len(length(y_train))
hs_train    <- sin(2 * pi * trend_index / freq)
hc_train    <- cos(2 * pi * trend_index / freq)

X_train <- cbind(hs = hs_train, hc = hc_train)

# --------------------------------------------------------------------------
# 4. Model specification (reduced model from the vignette)
# --------------------------------------------------------------------------
ar_lags <- c(10, 18)
ma_lags <- c(1, 13)
link    <- "logit"

# --------------------------------------------------------------------------
# 5. Start values at the 0.8 training slice
# --------------------------------------------------------------------------
start_par <- start_values(
  y    = y_train,
  link = link,
  ar   = ar_lags,
  ma   = ma_lags,
  xreg = X_train
)

cat("\nStart values:\n")
print(round(start_par, 6))

# --------------------------------------------------------------------------
# 6. Index map into start_par
# --------------------------------------------------------------------------
n_ar_params   <- length(ar_lags)
n_ma_params   <- length(ma_lags)
n_beta_params <- ncol(X_train)

idx_alpha  <- 1
idx_varphi <- seq.int(2, 1 + n_ar_params)
idx_theta  <- seq.int(2 + n_ar_params, 1 + n_ar_params + n_ma_params)
idx_beta   <- seq.int(2 + n_ar_params + n_ma_params,
                      1 + n_ar_params + n_ma_params + n_beta_params)
idx_phi    <- length(start_par)   # phi is last

# --------------------------------------------------------------------------
# 7. Log-likelihood at the start values  (16-digit reference)
# --------------------------------------------------------------------------
loglik_value <- loglik_barma(
  y      = y_train,
  ar     = ar_lags,
  ma     = ma_lags,
  alpha  = start_par[idx_alpha],
  varphi = start_par[idx_varphi],
  theta  = start_par[idx_theta],
  beta   = start_par[idx_beta],
  phi    = start_par[idx_phi],
  xreg   = X_train,
  link   = link,
  penalty = FALSE
)

cat("\nloglik_value :\n")
print(loglik_value, digits = 16)

# --------------------------------------------------------------------------
# 8. Score vector at the start values  (16-digit reference)
# --------------------------------------------------------------------------
score_value <- score_vector_barma(
  y      = y_train,
  ar     = ar_lags,
  ma     = ma_lags,
  alpha  = start_par[idx_alpha],
  varphi = start_par[idx_varphi],
  theta  = start_par[idx_theta],
  beta   = start_par[idx_beta],
  phi    = start_par[idx_phi],
  xreg   = X_train,
  link   = link,
  penalty = FALSE
)

cat("\nscore_value (order: alpha, varphi, theta, beta, phi):\n")
print(score_value, digits = 16)

# --------------------------------------------------------------------------
# 9. Persist everything as CSV (inputs AND expected outputs)
# --------------------------------------------------------------------------

# 9b. Expected outputs: scalar log-likelihood + score vector, 
# with a tagged layout so the Python fixture can read each piece 
# unambiguously.

score_names <- c(
  "alpha",
  paste0("varphi", ar_lags),
  paste0("theta",  ma_lags),
  colnames(X_train),
  "phi"
)

# Run the model
fit_cmle_reduced <- barma(
  y       = y_train,
  ar      = ar_lags,
  ma      = ma_lags,
  penalty = FALSE,
  xreg    = X_train
)

# Reference values
reference_data <- data.frame(
  component = score_names,
  score_value = as.numeric(score_value),
  estimates = as.numeric(coef(fit_cmle_reduced))
)

loglik_value_data <- data.frame(
  metric = "loglik",
  value = loglik_value
)

# --------------------------------------------------------------------------
# Export the CSV file
# --------------------------------------------------------------------------

write.csv(
  data.frame(param = names(start_par), 
             x = unname(start_par)),
  file.path(out_dir, "start_values_data.csv"),
  row.names = FALSE
)

write.csv(
  reference_data,
  file.path(out_dir, "reference_data.csv"),
  row.names = FALSE
)

write.csv(
  loglik_value_data,
  file.path(out_dir, "loglik_value_data.csv"),
  row.names = FALSE
)

write.csv(
  y_train,
  file.path(out_dir, "y_train_data.csv"),
  row.names = FALSE
)

write.csv(
  X_train,
  file.path(out_dir, "X_train_data.csv"),
  row.names = FALSE
)

