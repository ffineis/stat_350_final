# ------------------------------------- #
# Script to send data through an xgboost model
# resulting in a high-dimensional embedding of the dataset.
#
# E.g. `$ Rscript embed_xgboost_data.R -i ../data/train_5mil_processed.csv -m ../data/caret_xgbtree_Feb_19_2018_21-21-56.rds -c 1000000 -n 16 -o ../data/5mil_embed.rds
# ------------------------------------- #
require(data.table, quietly = TRUE)
require(optparse, quietly = TRUE)
require(fbboost, quietly = TRUE)


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
    , make_option(c('-m', '--model')
    			  , default = NULL
    			  , help = 'path to caret/xgboost model .RDS file')
	, make_option(c('-n', '--ncores')
				  , default = 2
				  , help = 'number of cores to run embedding procedure on')
    , make_option(c('-o', '--output_file')
                  , default = NULL
                  , help = 'filepath to where you would like to save embedded data as an .RDS')
    )

opt <- parse_args(OptionParser(option_list=optionList))

cat('LOADING DATA\n')
DT <- fread(opt$input
  , nrow = opt$train_count)
if('click' %in% names(DT)){
  DT[, click := NULL]
}

caretObj <- readRDS(opt$model)
xgbModel <- caretObj$finalModel


# ------------------------------------- #
# 2. Embed the XGbooster
# ------------------------------------- #
dat <- EmbedBooster(DT
                    , model = xgbModel
                    , nJobs = opt$ncores)

cat('EMBEDDED DATA SHAPE: ', dim(dat['data'])[1], dim(dat['data'])[2], '\n')


# ------------------------------------- #
# 3. Save training results, exit.
# ------------------------------------- #
cat('SAVING RESULTS TO ', opt$output_file, '\n')
saveRDS(dat
        , file = file.path(opt$output_file))

cat('\nSUCCESS. EXITING.\n\n')



