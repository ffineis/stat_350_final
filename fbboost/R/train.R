
#' @name WrapXgbTrain
#' @description Simple wrapper for xgboost::xgb.train
#' @param x input data (matrix, data.frame, or data.table). Shape is n x p.
#' @param y input labels. Must be n labels. Must be binary 0/1 integer class.
#' @param ... argumens to xgboost::xgb.train
#' @return xgb.Booster model
WrapXgbTrain <- function(x, y, ...){
  if(length(y) != dim(x)[1]){
    stop('number of rows of x and length of y do not conform.')
  }
  
  if(!class(y) %in% c('numeric', 'integer')){
    stop('y must be an integer vector or list of 0s and 1s')
  }
  args <- list(...)
  dat <- xgboost::xgb.DMatrix(as.matrix(x), label = y)
  args[['data']] <- dat
  
  bst <- do.call(xgboost::xgb.train, args)
  return(bst)
}

X <- data.table::as.data.table(matrix(rnorm(100), nrow = 20))
Y <- sample(c(0, 1), 20, replace = TRUE)
params <- list(max_depth = 2, eta = 1, silent = 1, nthread = 2,
                objective = "binary:logistic", eval_metric = "auc")
nrounds <- 2
model <- WrapXgbTrain(X, Y, params = params, nrounds = 2)
