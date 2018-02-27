#' @title Validate input passed to xgb.train
#' @name ValidateXgboostInput
#' @param inputNames names of params being sent as kwargs to do.call(xgboost, ...)
#' @description Make sure we're passing in valid input into xgb.train
ValidateXgboostInput <- function(inputNames){
	validInputNames <- c('data'
		, 'params'
		, 'nrounds'
		, 'watchlist'
		, 'obj'
		, 'feval'
		, 'verbose'
		, 'print_every_n'
		, 'early_stopping_rounds'
		, 'maximize'
		, 'save_period'
		, 'save_name'
		, 'xgb_model'
		, 'callbacks')

	invalidNames <- setdiff(inputNames, validInputNames)
	if(length(invalidNames) > 0){
		stop(paste0('Supplied arguments to xgb.train are not valid: '
			, paste0(inputNames, collapse = ', ')))
	} else {
		return(TRUE)
	}
}


#' @name GetDefaultLevels
#' @description Default levels of data for CTR marketing data.
#' For use with a target vector whose classes are 'pass' and 'click', e.g.
#' in web advertising datasets.
GetDefaultLevels <- function(){
	return(c('pass', 'click'))
}