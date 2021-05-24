#' Two-Stage-Least_Squares forest
#'
#' Extends previous codes of the grf-package to a two-stage least squares forest 
#'
#' @param X The covariates used in the instrumental regression.
#' @param Y The outcome.
#' @param W The treatment assignment
#' @param Z The instruments 
#' @param Y.hat Estimates of the expected responses E[Y | Xi], marginalizing
#'              over treatment. If Y.hat = NULL, these are estimated using
#'              a separate regression forest. Default is 0
#' @param W.hat Estimates of the treatment propensities E[W | Xi]. If W.hat = NULL,
#'              these are estimated using a separate regression forest. Default is 0
#' @param Z.hat Estimates of the expected instruments E[Z | Xi]. If Z.hat = NULL,
#'              these are estimated using a separate regression forest. Default is 0
#' @param num.trees Number of trees grown in the forest. Note: Getting accurate
#'                  confidence intervals generally requires more trees than
#'                  getting accurate predictions. Default is 2000.
#' @param sample.weights Not used
#' @param clusters Not used
#' @param sample.fraction Fraction of the data used to build each tree.
#'                        Note: If honesty = TRUE, these subsamples will
#'                        further be cut by a factor of honesty.fraction. Default is 0.5.
#' @param mtry Number of variables tried for each split. Default is NULL.
#' @param min.node.size A target for the minimum number of observations in each tree leaf. Note that nodes
#'                      with size smaller than min.node.size can occur, as in the original randomForest package.
#'                      Default is NULL.
#' @param honesty Whether to use honest splitting (i.e., sub-sample splitting). Default is TRUE.
#'  For a detailed description of honesty, honesty.fraction, and recommendations for
#'  parameter tuning, see the grf algorithm reference.
#' @param honesty.fraction The fraction of data that will be used for determining splits if honesty = TRUE. Corresponds
#'                         to set J1 in the notation of the paper. Default is 0.5 (i.e. half of the data is used for
#'                         determining splits).
#' @param alpha A tuning parameter that controls the maximum imbalance of a split. Default is NULL.
#' @param imbalance.penalty A tuning parameter that controls how harshly imbalanced splits are penalized. Default is NULL.
#' @param ci.group.size The forst will grow ci.group.size trees on each subsample.
#'                      In order to provide confidence intervals, ci.group.size must
#'                      be at least 2. Default is 2.
#' @param tune.parameters If TRUE, all paramters that are NULL are tuned.
#' @param tune.num.trees The number of trees in each 'mini forest' used to fit the tuning model.
#' @param tune.num.reps The number of forests used to fit the tuning model.
#' @param tune.num.draws The number of random parameter values considered when using the model
#'                          to select the optimal parameters. 
#' @param compute.oob.predictions Whether OOB predictions on training set should be precomputed. Default is TRUE.
#' @param num.threads Number of threads used in training. By default, the number of threads is set
#'                    to the maximum hardware concurrency.
#' @param seed The seed of the C++ random number generator.
#'
#' @return A trained two stage least squares forest object.
#'
#' @export
#' @useDynLib grftsls
#' @importFrom Rcpp evalCpp
#' 

tsls_forest <- function(X, Y, W, Z,
                        Y.hat,
                        W.hat,
                        Z.hat,
                        sample.fraction = NULL,
                        mtry = NULL,
                        num.trees = 2000,
                        min.node.size = NULL,
                        honesty = TRUE,
                        honesty.fraction = 0.5,
                        ci.group.size = 2,
                        alpha = NULL,
                        imbalance.penalty = NULL,
                        clusters = NULL,
                        samples.per.cluster = NULL,
                        tune.parameters = TRUE,
                        num.fit.trees = 10,
                        num.fit.reps = 100,
                        num.optimize.reps = 1000,
                        compute.oob.predictions = TRUE,
                        num.threads = NULL,
                        seed = NULL) {
  
  sample.weights <- NULL   
  
  # validate data
  validate_X(X)
  validate_sample_weights(sample.weights, X)
  Y = validate_observations(Y, X)
  W = validate_observations(W, X) 
  Z = validate_Z(Z,X)
  
  # validate parameters that are nict unted
  num.threads <- validate_num_threads(num.threads)
  seed <- validate_seed(seed)
  clusters <- validate_clusters(clusters, X)
  samples.per.cluster <- validate_samples_per_cluster(samples.per.cluster, clusters)
  honesty.fraction <- validate_honesty_fraction(honesty.fraction, honesty)
  
  # center
  args.orthog = list(X = X,
                     num.trees = min(500, num.trees),
                     sample.weights = sample.weights,
                     clusters = clusters,
                     equalize.cluster.weights = FALSE,
                     sample.fraction = NULL,
                     mtry = NULL,
                     min.node.size = NULL,
                     honesty = TRUE,
                     honesty.fraction = 0.5,
                     honesty.prune.leaves = TRUE,
                     alpha = NULL,
                     imbalance.penalty = NULL,
                     ci.group.size = 1,
                     tune.parameters = "all",  
                     num.threads = num.threads,
                     seed = seed)
  
  if (is.null(Y.hat)) {
    Y.hat <- do.call(grf::regression_forest, c(Y = list(Y), args.orthog))$predictions
  } else if (length(Y.hat) == 1) {
    Y.hat <- rep(Y.hat, nrow(X))
  } else if (length(Y.hat) != nrow(X)) {
    stop("Y.hat has incorrect length.")
  }
  
  if (is.null(W.hat)) {
    W.hat <- do.call(grf::regression_forest, c(Y = list(W), args.orthog))$predictions
  } else if (length(W.hat) == 1) {
    W.hat <- rep(W.hat, nrow(X))
  } else if (length(W.hat) != nrow(X)) {
    stop("W.hat has incorrect length.")
  }

  if (is.null(Z.hat)) {
    Z.hat = matrix(nrow = nrow(X), ncol = ncol(Z))
    for(i in seq_len(ncol(Z))) {
      Z.hat[,i] <- do.call(grf::regression_forest, c(Y = list(Z[,i]), args.orthog))$prediction
    }
  } else if (length(Z.hat) == 1) {
    Z.hat  <- matrix(rep(Z.hat, nrow(X)*ncol(Z)), nrow=nrow(X), ncol=ncol(Z))
  } else if (is.vector(Z.hat)) {
    if(length(Z.hat != ncol(Z))) {
      stop("Z.hat has not the same number of columns as Z")
    } else {
      Z.hat  <- matrix(rep(Z.hat, nrow(X)), nrow=nrow(X), ncol = ncol(Z), byrow = TRUE)
    }  
  } else if (ncol(Z.hat) != ncol(Z) | nrow(Z.hat)!=nrow(X)) {
    stop("Z has the wrong number of rows and/or columns")
  }

  # tune
  if (tune.parameters) {
    # if NULL,  tune parameter, if not NULL do not tune it.
    tuning.output <- tune_tsls_forest(X, Y, W, Z, Y.hat, W.hat, Z.hat,
                                      sample.weights = sample.weights,
                                      num.fit.trees = num.fit.trees,
                                      num.fit.reps = num.fit.reps,
                                      num.optimize.reps = num.optimize.reps,
                                      min.node.size = min.node.size,
                                      sample.fraction = sample.fraction,
                                      mtry = mtry,
                                      alpha = alpha,
                                      imbalance.penalty = imbalance.penalty,
                                      num.threads = num.threads,
                                      honesty = honesty,
                                      honesty.fraction = honesty.fraction,
                                      seed = seed,
                                      clusters = clusters,
                                      samples.per.cluster = samples.per.cluster)
    tunable.params <- tuning.output$params
  } else {
    tunable.params <- c(
      min.node.size = validate_min_node_size(min.node.size),
      sample.fraction = validate_sample_fraction(sample.fraction),
      mtry = validate_mtry(mtry, X),
      alpha = validate_alpha(alpha),
      imbalance.penalty = validate_imbalance_penalty(imbalance.penalty))
  }
  
  data <- create_data_matrices(X, Y - Y.hat, W - W.hat, Z - Z.hat, sample.weights = sample.weights)
    
  outcome.index <- ncol(X) + 1
  treatment.index <- ncol(X) + 2
  instrument.index <- seq(from = ncol(X) + 3, to = ncol(X) + 3 + ncol(Z)-1 ,by = 1)
  sample.weight.index <- ncol(X) + 4
  
  instrument.index <- instrument.index - 1 # C++ format!
  forest <- tsls_train(data$default, data$sparse,
                       outcome.index, treatment.index, instrument.index, sample.weight.index,
                       !is.null(sample.weights),
                       as.numeric(tunable.params["mtry"]),
                       num.trees,
                       as.numeric(tunable.params["min.node.size"]),
                       as.numeric(tunable.params["sample.fraction"]),
                       honesty,
                       coerce_honesty_fraction(honesty.fraction),
                       ci.group.size,
                       as.numeric(tunable.params["alpha"]),
                       as.numeric(tunable.params["imbalance.penalty"]),
                       clusters,
                       samples.per.cluster,
                       compute.oob.predictions,
                       num.threads,
                       seed)
  
  
  class(forest) <- c("tsls_forest", "grf")
  forest[["ci.group.size"]] <- ci.group.size
  forest[["X.orig"]] <- X
  forest[["Y.orig"]] <- Y
  forest[["W.orig"]] <- W 
  forest[["Z.orig"]] <- Z
  forest[["Y.hat"]] <- Y.hat
  forest[["W.hat"]] <- W.hat
  forest[["Z.hat"]] <- Z.hat
  forest[["tunable.params"]] <- tunable.params
  forest[["sample.weights"]] <- sample.weights
  forest[["clusters"]] <- clusters
  
  forest
}

#' Predict with a two-stage-least-squares forest
#'
#' @method predict tsls_forest
#' @export
predict.tsls_forest <- function(object, newdata = NULL,
                                      num.threads = NULL,
                                      estimate.variance = TRUE,
                                      ...) {

  # If possible, use pre-computed predictions.
  if (is.null(newdata) & !estimate.variance & !is.null(object$predictions)) {
    return(data.frame(predictions=object$predictions,
                      debiased.error=object$debiased.error,
                      excess.error=object$excess.error))
  }

  num.threads = validate_num_threads(num.threads)

  forest.short = object[-which(names(object) == "X.orig")]
  X = object[["X.orig"]]
  Y.centered = object[["Y.orig"]] - object[["Y.hat"]]
  W.centered = object[["W.orig"]] - object[["W.hat"]]
  Z.centered = object[["Z.orig"]] - object[["Z.hat"]]

  train.data <- create_data_matrices(X, Y.centered, W.centered, Z.centered)

  outcome.index = ncol(X) + 1
  treatment.index <- ncol(X) + 2
  instrument.index <- seq(from = ncol(X) + 3, to = ncol(X) + 3 + ncol(Z)-1 ,by = 1)

  if (!is.null(newdata) ) {
    validate_newdata(newdata, object$X.orig)
    data <- create_data_matrices(newdata)
    ret = tsls_predict(forest.short, train.data$default, train.data$sparse, outcome.index,
                             treatment.index, instrument.index, data$default, data$sparse,
                             num.threads, estimate.variance)
  } else {
    ret = tsls_predict_oob(forest.short, train.data$default, train.data$sparse, outcome.index,
                             treatment.index, instrument.index, num.threads, estimate.variance)
  }

  # Convert list to data frame.
  empty = sapply(ret, function(elem) length(elem) == 0)
  do.call(cbind.data.frame, ret[!empty])
}





