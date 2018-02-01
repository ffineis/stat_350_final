# ---- Load reqd packages
library(testthat)
library(data.table)

# ---- Clear environment
rm(list = ls())


# ------------------ #
# ---- OVERHEAD ---- #
# ------------------ #
irisDT <- as.data.table(iris)
binaryClassDT <- copy(irisDT)
multiClassDT <- copy(irisDT)

binaryClassDT[, Species := ifelse(as.character(Species) == 'setosa', 1, 0)]
multiClassDT[, Species := unlist(lapply(as.character(Species), FUN = function(x){
	switch(x
		, setosa = 0
		, versicolor = 1
		, virginica = 2)
	}))]

binaryClassParams <- list(max_depth = 2
	, eta = 1
	, silent = 1
	, nthread = 2
	, colsample_bytree = 0.8
	, booster = 'gbtree'
	, objective = 'binary:logistic'
	, eval_metric = 'auc')

multiClassParams <- list(max_depth = 2
	, eta = 1
	, silent = 1
	, num_class = 3
	, nthread = 2
	, colsample_bytree = 0.8
	, booster = 'gbtree'
	, objective = 'multi:softmax'
	, eval_metric = 'map')

varNames <- setdiff(names(irisDT), 'Species')
nRounds <- 3


# --------------- #
# ---- TESTS ---- #
# --------------- #

# ---- CONTEXT: xgb.train wrapper tests
context('WrapXgbTrain testing...')

test_that('WrapXgbTrain can work with y being a vector', {
	bst <- WrapXgbTrain(x = binaryClassDT[, .SD, .SDcols = varNames]
		, y = binaryClassDT[, Species]
		, params = binaryClassParams
		, nrounds = nRounds)

	expect_true(class(bst) == 'xgb.Booster')
})

test_that('WrapXgbTrain fails when y is not the name of a field of x', {
	expect_error(WrapXgbTrain(x = binaryClassDT
			, y = 'not-a-valid-field-name'
			, params = binaryClassParams
			, nrounds = nRounds)
		, 'y is not a valid name of a variable in x')
})

test_that('WrapXgbTrain can work with multiclass labels', {
	bst <- WrapXgbTrain(x = multiClassDT
		, y = 'Species'
		, params = multiClassParams
		, nrounds = nRounds)

	expect_true(class(bst) == 'xgb.Booster')
})


# ---- Clean up
rm(list = ls())
