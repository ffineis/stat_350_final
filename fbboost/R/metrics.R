#' @title Normalized entropy calculator
#' @name NormalizedEntropy
#' @description Calculate normalized entropy, i.e. binary
#' cross entropy divided by entropy of background probability of positive class label;
#' defined in He, Pan, et al 2014
#' @param trueVec numeric vector of true class labels
#' @param predVec numeric vector of class probability prediction
#' @return numeric value
#' @export
NormalizedEntropy <- function(trueVec, predVec){
  if(length(trueVec) != length(predVec)){
    stop('Actual and predicted vectors need to have the same length')
  }
  
  crossEntropy <- -((1 + trueVec) / 2)*log(predVec) - ((1 - trueVec) / 2)*log(1 - predVec)
  ctr <- mean(trueVec == 1)
  ctrEntropy <- -ctr*log(ctr) - (1 - ctr)*log(1 - ctr)
  return(mean(crossEntropy) / ctrEntropy)
}

#' @title Calculate Normalized Entropy across samples
#' @name NormalizedEntropySummary
#' @description summary function to use Normalized Entropy metric in caret::train
#' see ?caret::defaultSummary
#' @param data data.table provided by caret::train
#' @param lev str positive class label
#' @param model NULL required by caret::train
#' @return named list
#' @export
NormalizedEntropySummary <- function(data, lev = GetDefaultLevels()[2], model = NULL){
  levels(data$obs) <- c('-1', '1')
  out <- NormalizedEntropy(as.numeric(levels(data$obs))[data$obs]
                           , predVec = data[, lev])
  names(out) <- 'NormalizedEntropy'
  return(out)
}