#' @title xgboost::xgb.train wrapper
#' @name WrapXgbTrain
#' @description Simple wrapper for xgboost::xgb.train
#' @param x input data (matrix, data.frame, or data.table). Shape is n x p.
#' @param y input labels or name of target variable present in x.
#' @param ... arguments to xgboost::xgb.train
#' @return xgb.Booster model
#' @importFrom xgboost xgb.train xgb.DMatrix
WrapXgbTrain <- function(x, y, ...){
  
  # Handle case when y is name of a field in x.
  if(length(y) == 1){
    xFields <- names(x)
    
    if((class(y) == 'character') && (y %in% xFields)){
      target <- as.numeric(x[[y]])
      xFields <- setdiff(xFields, y)
      
      if('data.table' %in% class(x)){
        input <- x[, .SD, .SDcols = xFields]
      } else {
        input <- x[, xFields]
      }
    } else {
      stop('y is not a valid name of a variable in x')
    }
  # Handle case when y is actually a vector of integer/numeric vals
  } else if((length(y) > 1) && (class(y) %in% c('numeric', 'integer'))){
    if(length(y) != dim(x)[1]){
      stop('number of rows of x and length of y do not conform.')
    }
    target <- as.numeric(y)
    input <- x
  } else {
    stop('y must be a numeric/integer vector of class labels or the name of the target variable in x')
  }
  
  args <- list(...)
  dat <- xgboost::xgb.DMatrix(as.matrix(input), label = target)
  args[['data']] <- dat
  val <- ValidateXgboostInput(names(args))
  
  bst <- do.call(xgboost::xgb.train, args)
  return(bst)
}
