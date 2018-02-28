# ------------------------------------- #
# Script to train a logistic regression classifier model
#
# E.g. `$Rscript train_logistic_model.R --x_input ../data/500k_embed.RDS --y_input ../data/train_500k_processed.csv -c 300000 -n 2 -k 5 -a 1 --nlambda 20 -o ../data/lasso_glmnet.rds
# ------------------------------------- #
require(data.table, quietly = TRUE)
require(optparse, quietly = TRUE)
require(glmnet, quietly = TRUE)
require(caret, quietly = TRUE)


# ------------------------------------- #
# 1. Parse cmd line arguments, load/format data
# ------------------------------------- #
optionList <- list( 
    make_option(c('-x', '--x_input')
                  , default = '../data/10k_embed.RDS'
                  , help = 'path to X data')
    , make_option(c('-y', '--y_input')
                  , default = '../data/train_500k_processed.csv'
                  , help = 'path to Y target data file. File must contain `click` field.')
    , make_option(c('-m', '--method')
                  , default = 'glmnet'
                  , help = 'Model training method. Either `glmnet` or `caret`. If `caret`, NormalizedEntropy metric is used.')
    , make_option(c('-c', '--train_count')
    			        , default = 10000
    			        , help = 'first train_count-many observations will be used for model training, remaining for test set')
	  , make_option(c('-n', '--ncores')
				          , default = 2
				          , help = 'number of cores to run training procedure on')
    , make_option(c('-k', '--kfolds')
                  , default = 3
                  , help = 'number of k fold validation folds to use')
    , make_option(c('-a', '--alpha')
                  , default = 1
                  , help = 'alpha elasticnet l1-l2 balance coefficient')
    , make_option(c('-l', '--nlambda')
                  , default = 10
                  , help = 'number of lambda values to use to train elasticnet')
    , make_option(c('-o', '--output_file')
                  , default = NULL
                  , help = 'filepath to where you would like to save the elasticnet model and predictions as an .RDS')
    )

opt <- parse_args(OptionParser(option_list=optionList))

if(!opt$method %in% c('glmnet', 'caret')){
  stop('method argument must be either `glmnet` or `caret`.')
}

if(opt$ncores > parallel::detectCores()){
  stop('ncores cannot be greater than ', parallel::detectCores())
}

fileEncoding <- tolower(substr(opt$x_input
                        , start = nchar(opt$x_input)-3
                        , stop = nchar(opt$x_input)))

cat('LOADING X DATA FROM ', opt$x_input, '\n')
if(fileEncoding == '.rds'){
  x <- readRDS(opt$x_input)$data
} else if(fileEncoding == '.csv'){
  DT <- fread(opt$x_input)
  if('click' %in% colnames(DT)){
    DT[, click := NULL]
  }
  x <- as.matrix(DT)
}
if(is.null(colnames(x))){
  colnames(x) <- paste0('X', 1:ncol(x))
}

y <- data.table::fread(opt$y_input
                       , select = 'click')$click


# ------------------------------------- #
# 2. Prepare data
# ------------------------------------- #
cat('DROPPING 0-VARIANCE FEATURES\n')
colVars <- apply(x
                 , MARGIN = 2
                 , FUN = var)
x <- x[, setdiff(1:ncol(x), which(colVars == 0))]

if(opt$train_count > nrow(x)){
  stop('train_count argument cannot exceed number of rows in x')
}

trainIdx <- 1:opt$train_count
if(opt$train_count == nrow(x)){
  warning('train_count was set to nrow(x). No test set is provided.')
  testIdx <- NULL
} else {
  testIdx <- c((opt$train_count + 1):nrow(x))
}

y <- as.factor(y)
levels(y) <-c('pass', 'click') 


# ------------------------------------- #
# 3. Kick of parallel cluster, train
# ------------------------------------- #
cat('INITIALIZING PARALLEL CLUSTER\n')
cluster <- parallel::makeCluster(opt$ncores)
doParallel::registerDoParallel(cluster)
out <- parallel::clusterEvalQ(cluster, library("fbboost"))

if(opt$method == 'glmnet'){
  cat('TRAINING MODEL WITH', opt$nlambda, 'lambdas;', 'alpha =', opt$alpha, '\n')
  cvFit <- cv.glmnet(x[trainIdx, ]
                    , y = y[trainIdx]
                    , nfolds = opt$kfolds
                    , alpha = opt$alpha
                    , nlambda = opt$nlambda
                    , parallel = TRUE
                    , standardize = FALSE
                    , family = 'binomial')
} else {
  trControl <- caret::trainControl(method = 'cv'
                                   , number = opt$kfolds
                                   , verboseIter = TRUE
                                   , classProbs = TRUE
                                   , summaryFunction = fbboost::NormalizedEntropySummary
                                   , allowParallel = TRUE)

  tuneGrid <- expand.grid(alpha = opt$alpha
                          , lambda = seq(0.001
                                         , to = 2
                                         , length.out = opt$nlambda))
  
  cvFit <- caret::train(x = x[trainIdx, ]
                        , y = y[trainIdx]
                        , method = 'glmnet'
                        , metric = 'NormalizedEntropy'
                        , trControl = trControl)
}

parallel::stopCluster(cluster)


# ------------------------------------- #
# 4. Get class membership probabilities
# ------------------------------------- #
if(!is.null(testIdx)){
  cat('OBTAINING TEST SET PREDICTED CLASS PROBABILITIES\n')
  if(opt$method == 'glmnet'){
    preds <- predict(cvFit
                     , newx = x[testIdx, ]
                     , s = 'lambda.min'
                     , type = 'response')
  } else {
    preds <- predict(cvFit
                     , x[testIdx, ])
  }
} else {
  preds <- NULL
}


# ------------------------------------- #
# 5. Save fitted model and predictions
# ------------------------------------- #
cat('SAVING RESULTS TO', opt$output_file, '\n')
outList <- list('model' = cvFit
                , 'preds' = preds)
saveRDS(outList
        , file = opt$output_file)







