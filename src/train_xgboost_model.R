# ------------------------------------- #
# Script to train xgboost model on processed Avazu CTR data,
# using normalized entropy as loss function.
# Save resulting trained model (and metrics from caret).
#
# E.g. `$ Rscript train_xgboost_model.R -i ../data/train_5mil_processed.csv -c 3000000 -n 3`
# ------------------------------------- #
require(data.table, quietly = TRUE)
require(optparse, quietly = TRUE)
require(caret, quietly = TRUE)
require(parallel, quietly = TRUE)
require(doParallel, quietly = TRUE)


# ------------------------------------- #
# 1. Parse cmd line arguments, load/format data
# ------------------------------------- #
optionList <- list( 
    make_option(c('-i', '--input')
                , default = '../data/train_5mil_processed.csv'
                , help = 'full filepath to input Avazu Kaggle CTR data')
    , make_option(c('-c', '--train_count')
    			  , default = 10000
    			  , help = 'first train_count-many observations will be used for model training')
    , make_option(c('-k', '--kfolds')
    			  , default = 3
    			  , help = 'number of k fold validation folds to use')
	, make_option(c('-n', '--ncores')
				  , default = 2
				  , help = 'number of cores to run training on')
    , make_option(c('-o', '--output_dir')
                  , default = file.path('..', 'data')
                  , help = 'full filepath of processed output .csv datafile, e.g. ./ctr_output.csv')
    )

opt <- parse_args(OptionParser(option_list=optionList))

cat('LOADING DATA\n')
DT <- fread(opt$input)

# (negative label, positive label) factor levels (caret::train uses factors for target)
LEVELS <- c('pass', 'click')

# Obtain training/test index split
trainIdx <- c(1:opt$train_count)
testIdx <- setdiff(1:nrow(DT), trainIdx)

# format X, Y data
y <- as.factor(DT[, get('click')])
levels(y) <- LEVELS
DT <- DT[, click := NULL]


# ------------------------------------- #
# 2. Define NormalizedEntropy for caret
# ------------------------------------- #
NormalizedEntropy <- function(actualVec, predVec){
  if(length(actualVec) != length(predVec)){
    stop('Actual and predicted vectors need to have the same length')
  }
  
  crossEntropy <- -((1+actualVec)/2)*log(predVec) - ((1-actualVec)/2)*log(1-predVec)
  ctr <- mean(actualVec == 1)
  ctrEntropy <- -ctr*log(ctr) - (1-ctr)*log(1-ctr)
  return(mean(crossEntropy) / ctrEntropy)
}

NormalizedEntropySummary <- function(data, lev = LEVELS[2], model = NULL){
  levels(data$obs) <- c('-1', '1')
  out <- NormalizedEntropy(as.numeric(levels(data$obs))[data$obs]
                           , predVec = data[, LEVELS[2]])
  names(out) <- 'NormalizedEntropy'
  return(out)
}


# ------------------------------------- #
# 3. Define caret/xgboost parameters
# ------------------------------------- #
trControl <- caret::trainControl(method = 'cv'
                                 , number = opt$kfolds
                                 , verboseIter = TRUE
                                 , classProbs = TRUE
                                 , summaryFunction = NormalizedEntropySummary
                                 , allowParallel = TRUE)

tuneGrid <- expand.grid(nrounds = c(50, 125, 200)
                        , max_depth = c(3, 5, 7)
                        , eta = c(0.2)
                        , gamma = c(1)
                        , colsample_bytree = c(0.7)
                        , subsample = c(0.5)
                        , min_child_weight = c(1))


# ------------------------------------- #
# 4. Set up cluster, run caret::train
# ------------------------------------- #
cat('ESTABLISHING CLUSTER FOR caret::train PARALLEL EXECUTION\n')
cl <- makeCluster(opt$ncores)
registerDoParallel(cl)
out <- lapply(list('NormalizedEntropy', 'NormalizedEntropySummary', 'LEVELS')
			  , FUN = function(x){clusterExport(cl, x)})

cat('TRAINING/CROSS VALIDATING MODEL WITH ', nrow(tuneGrid), 'UNIQUE PARAMETER SETTINGS\n')
cvResults <- caret::train(DT[trainIdx]
                          , y = y[trainIdx]
                          , method = 'xgbTree'
                          , metric = 'NormalizedEntropy'
                          , trControl = trControl
                          , tuneGrid = tuneGrid
                          , maximize = FALSE)
stopCluster(cl)


# ------------------------------------- #
# 5. Get idea of model performance
# ------------------------------------- #
clickPreds <- predict(cvResults
                      , newdata = DT[testIdx])
acc <- mean(clickPreds == y[testIdx])
tpr <- mean(y[testIdx][which(clickPreds == LEVELS[2])] == LEVELS[2])
cat('TEST SET METRICS:\n\tAccuracy =', acc, ' ---- true positive rate:', tpr, '\n')


# ------------------------------------- #
# 6. Save training results, exit.
# ------------------------------------- #
modelFileName <- paste0('caret_xgbtree_'
                        , format(Sys.time(), "%b_%d_%Y_%H-%M-%S")
                        , '.rds')
cat('SAVING RESULTS TO ', modelFileName, '\n')
saveRDS(cvResults
        , file = file.path(opt$output
                           , modelFileName))

cat('\nSUCCESS. EXITING.\n\n')



