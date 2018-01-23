# ------------------------------------- #
# Script for processing raw Avazu click 
# through rate data from Kaggle. Save to a .csv file.
#
# E.g. `$ Rscript preprocess_avazu_data.R -i train_5mil.csv -o train_5mil_processed.csv`
# ------------------------------------- #

require(data.table, quietly = TRUE)
require(optparse, quietly = TRUE)
require(lubridate, quietly = TRUE)
require(feather, quietly = TRUE)

# continuous variables
CTSVARS <- c('C14'
             , 'C17'
             , 'C19'
             , 'C20'
             , 'C21'
             , 'hour'
             , 'day_of_week')

# categorical variables
CATVARS <- c('C1'
             , 'banner_pos'
             , 'site_category'
             , 'app_category'
             , 'device_type'
             , 'device_conn_type'
             , 'C15'
             , 'C16')

# variables to binarize
BINARIZEVARS <- c('site_id'
                  , 'site_domain'
                  , 'app_id'
                  , 'app_domain'
                  , 'device_id')

# target variable
YVAR <- 'click'

# map defining how to replace certain values in certain fields
REPLACE_MAP <- list()
REPLACE_MAP[['C20']] <- list()
REPLACE_MAP[['C20']][['-1']] <- NA_integer_


# ------------------------------------- #
# 1. Parse cmd line arguments, load data
# ------------------------------------- #
cat('BEGIN PROCESSING\n')
optionList <- list( 
    make_option(c('-i', '--input')
                , default = NULL
                , help = 'full filepath to input Avazu Kaggle CTR data'),
    make_option(c('-o', '--output')
                , default = paste0(format(Sys.time(), "%y%m%d%H%M%S"), '_processed_ctr.csv')
                , help = 'full filepath of processed output .csv datafile, e.g. ./ctr_output.csv')
    )

opt <- parse_args(OptionParser(option_list=optionList))

if(is.null(opt$input)){
  stop('`input` cmd line argument cannot be missing. Do not know which Avazu CTR data file you would like to process.')
} else {
  outputFormat <- substring(opt$output
                            , first = nchar(opt$output) - 3
                            , last = nchar(opt$output))
  if(outputFormat != '.csv'){
    stop(opt$output, 'is not a valid output file name. Must be a .csv file.')
  }
  cat('input Avazu CTR data filepath ----', opt$input, '\n')
  cat('output processed data filepath ----', opt$output, '\n')
}

DT <- fread(opt$input
            , colClasses = list(character = 'id'))

# ----------------------------------------- #
# 2. Transform time vars
# ----------------------------------------- #
cat('Transforming time variables...\n')
DT[, hour := as.character(hour)]
DT[, dateTime := as.POSIXct(get('hour')
                            , format = '%y%m%d%H'
                            , tz = 'UTC')]
DT[, hour := lubridate::hour(dateTime)]
DT[, day_of_week := lubridate::wday(dateTime)]
DT[, dateTime := NULL]


# ----------------------------------------- #
# 3. Binarize the brutally categorical vars
# ----------------------------------------- #
cat('Binarizing specified variables...\n')
for(col in BINARIZEVARS){
  tab <- table(DT[, get(col)])
  maxOcc <- tab[which.max(tab)]
  maxOccVal <- names(maxOcc)
  DT[, eval(col) := ifelse(get(col) == maxOccVal, 1, 0)]
}


# ----------------------------------------- #
# 4. Replace values with REPLACE_MAP
# ----------------------------------------- #
cat('Replacing specific values of specific variables...\n')
for(col in names(REPLACE_MAP)){
  replaceVals <- names(REPLACE_MAP[[col]])
  for(replaceVal in replaceVals){
    newVal <- REPLACE_MAP[[col]][[replaceVal]]
    DT[DT[, get(col)] == replaceVal
       , eval(col) := newVal]
  }
}


# ----------------------------------------- #
# 5. Standardize continuous vars
# ----------------------------------------- #
cat('Standardizing continuous variables...\n')
DT[, eval(CTSVARS) := lapply(.SD, FUN = function(x){
    mu <- mean(x, na.rm = TRUE)
    std <- sd(x, na.rm = TRUE)
    return((x-mu)/std)
  })
  , .SDcols = CTSVARS]


# ----------------------------------------- #
# 6. One-hot-encode categorical vars
# ----------------------------------------- #
cat('One-hot-encoding categorical variables...\n')
DT[, eval(CATVARS) := lapply(.SD, FUN = as.character)
   , .SDcols = CATVARS]
form <- formula(paste0(YVAR
                       , '~'
                       , paste0(CATVARS, collapse = ' + ')
                       , ' - 1'))
oheDT <- as.data.table(model.matrix(form
                                    , data = DT))


# ----------------------------------------- #
# 7. Write processed data to disk
# ----------------------------------------- #
cat('Writing data to', opt$output, '...\n\n', sep = '')

# data that's ready for modeling...?
outDT <- cbind(DT[, .SD, .SDcols = c(YVAR, CTSVARS, BINARIZEVARS)]
               , oheDT)
rm(oheDT)
rm(DT)

write.csv(outDT
          , file = opt$output)
cat('END PROCESSING\n')