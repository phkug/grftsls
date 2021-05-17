#' @importFrom stats runif
#' @importFrom utils capture.output
#' @useDynLib grftsls
#' @importFrom Rcpp evalCpp
#' @export
tune_tsls_forest <- function(X, Y, W, Z, Y.hat, W.hat, Z.hat,
                             sample.weights = NULL,
                             num.fit.trees = 200,
                             num.fit.reps = 50,
                             num.optimize.reps = 1000,
                             min.node.size = NULL,
                             sample.fraction = NULL,
                             mtry = NULL,
                             alpha = NULL,
                             imbalance.penalty = NULL,
                             honesty = TRUE,
                             honesty.fraction = NULL,
                             clusters = NULL,
                             samples.per.cluster = NULL,
                             num.threads = NULL,
                             seed = NULL,
                             max.sample.fraction = 0.5,
                             min.min.node.size = 1,
                             min.mtry = 1,
                             min.alpha = 0) {
  
  validate_X(X)
  validate_sample_weights(sample.weights, X)
  Y = validate_observations(Y, X)
  W = validate_observations(W, X) 
  Z = validate_Z(Z,X)
  
  num.threads <- validate_num_threads(num.threads)
  seed <- validate_seed(seed)
  clusters <- validate_clusters(clusters, X)
  samples.per.cluster <- validate_samples_per_cluster(samples.per.cluster, clusters)
  honesty.fraction <- validate_honesty_fraction(honesty.fraction, honesty)
  ci.group.size <- 1
  
  data <- create_data_matrices(X, Y - Y.hat, W - W.hat, Z - Z.hat, sample.weights = sample.weights)
  outcome.index <- ncol(X) + 1
  treatment.index <- ncol(X) + 2
  instrument.index <- seq(from = ncol(X) + 3, to = ncol(X) + 3 + ncol(Z)-1 ,by = 1)
  sample.weight.index <- ncol(X) + 4
  instrument.index <- instrument.index - 1 # C++ format!
  
  # Separate out the tuning parameters with supplied values, and those that were
  # left as 'NULL'. We will only tune those parameters that the user didn't supply.
  all.params = get_initial_params(min.node.size, sample.fraction, mtry, alpha, imbalance.penalty)
  fixed.params = all.params[!is.na(all.params)]
  tuning.params = all.params[is.na(all.params)]
  
  if (length(tuning.params) == 0) {
    return(list("error"=NA, "params"=c(all.params)))
  }
  
  # Train several mini-forests, and gather their debiased OOB error estimates.
  num.params = length(tuning.params)
  fit.draws = matrix(runif(num.fit.reps * num.params), num.fit.reps, num.params)
  colnames(fit.draws) = names(tuning.params)
  compute.oob.predictions = TRUE
  
  debiased.errors = apply(fit.draws, 1, function(draw) {
    params = c(fixed.params, get_params_from_draw(X, draw,
                                                  max.sample.fraction = max.sample.fraction, 
                                                  min.min.node.size = min.min.node.size-1,
                                                  min.mtry = min.mtry,
                                                  min.alpha = min.alpha))
    small.forest <- tsls_train(data$default, data$sparse,
                               outcome.index, treatment.index, instrument.index, sample.weight.index,
                               !is.null(sample.weights),
                               as.numeric(params["mtry"]),
                               num.fit.trees,
                               as.numeric(params["min.node.size"]),
                               as.numeric(params["sample.fraction"]),
                               honesty,
                               coerce_honesty_fraction(honesty.fraction),
                               ci.group.size,
                               as.numeric(params["alpha"]),
                               as.numeric(params["imbalance.penalty"]),
                               clusters,
                               samples.per.cluster,
                               compute.oob.predictions,
                               num.threads,
                               seed)
    prediction = tsls_predict_oob(small.forest, data$default, data$sparse,
                                  outcome.index,treatment.index, instrument.index, num.threads, FALSE)
    mean(prediction$debiased.error, na.rm = TRUE)
  })
  
  # Fit the 'dice kriging' model to these error estimates.
  # Note that in the 'km' call, the kriging package prints a large amount of information
  # about the fitting process. Here, capture its console output and discard it.
  variance.guess = rep(var(debiased.errors)/2, nrow(fit.draws))
  env = new.env()
  capture.output(env$kriging.model <-
                   DiceKriging::km(design = data.frame(fit.draws),
                                   response = debiased.errors,
                                   noise.var = variance.guess))
  kriging.model <- env$kriging.model
  
  # To determine the optimal parameter values, predict using the kriging model at a large
  # number of random values, then select those that produced the lowest error.
  optimize.draws = matrix(runif(num.optimize.reps * num.params), num.optimize.reps, num.params)
  colnames(optimize.draws) = names(tuning.params)
  model.surface = predict(kriging.model, newdata=data.frame(optimize.draws), type = "SK")
  
  tuned.params = get_params_from_draw(X, optimize.draws, 
                                      max.sample.fraction = max.sample.fraction, 
                                      min.min.node.size = min.min.node.size-1,
                                      min.mtry = min.mtry,
                                      min.alpha = min.alpha)
  grid = cbind(error=model.surface$mean, tuned.params)
  optimal.draw = which.min(grid[, "error"])
  optimal.param = grid[optimal.draw, ]
  
  out = list(error = optimal.param[1], params = c(fixed.params, optimal.param[-1]),
             grid = grid)
  class(out) = c("tuning_output")
  
  out
}
  
