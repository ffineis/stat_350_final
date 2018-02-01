#' @title Back out trees from boosted tree model from xgboost package
#' @name GetTreeTable
#' @description Make the output of xgboost::xgb.model.dt.tree more useful
#' @param featureNames names of data as it was sent into xgboost::xgb.train
#' @param model an xgb.Booster model
#' @return a data.table of trees
#' @importFrom xgboost xgb.model.dt.tree
#' @importFrom data.table :=
GetTreeTable <- function(featureNames, model){
  
  # param checking
  if(!('xgb.Booster' %in% class(model))){
    stop('model argument must be of class xgb.Booster')
  }
  if(length(featureNames) < 1){
    stop('length(featureNames) must be > 0')
  }
  
  # Get xgboost representation of tree learners in a booster model
  trees <- tryCatch({
    xgboost::xgb.model.dt.tree(featureNames
                               , model = model)
  }, error = function(e){
    stop('xgb.model.dt.tree failed to back out trees using supplied featureNames and model.')
  })
  
  # xgboost 0-indexes trees/node ids, so 1-index them.
  trees[, Tree := Tree + 1]
  trees[, Node := Node + 1]
  
  # change xgboost tree learner representation to something more intuitive
  for(field in c('ID', 'Yes', 'No', 'Missing')){
    trees[, eval(field) := unlist(lapply(get(field)
                                  , FUN = function(x){1 + as.integer(strsplit(x, '-')[[1]][2])}))]
  }
  
  # validate that we sucked out the right number of trees
  if(max(trees[, Tree]) != model$niter){
    stop('Number of treeds found does not equal number of rounds used to train Booster.')
  }

  return(trees)
}

#' @title Embed data in a single tree created from GetTreeTable
#' @name EmbedInTree
#' @description Send a datapoint through a decision tree, embedding the data in binary tree-leaf space, that tree
#' being a structure created with the GetTreeTable function.
#' @param x input data (matrix, data.frame, or data.table). Shape is n x p.
#' @param tree processed data.table output from xgboost::xgb.model.dt.tree
#' @return binary embedding matrix. Of shape n x #-terminal-nodes-in-tree
EmbedInTree <- function(x, tree){
  
  # param checking
  nObs <- dim(x)[1]
  if(nObs < 1){
    stop('Cannot embed < 1 observations in to tree space!')
  }
  if(!('data.table' %in% class(tree))){
    stop('tree parameter must be a data.table; use output from GetTreeTable.')
  }
  if(length(unique(tree[, Node])) != nrow(tree)){
    stop('Submit one tree at a time. tree parameter must have one row per node.')
  }
  
  # output storage
  embeddedMat <- matrix(0
                        , nrow = nObs
                        , ncol = max(tree[, Node]))
  
  # embed all observations in tree space
  for(i in 1:nObs){
    xVec <- x[i, ]
    node <- 1
    isLeaf <- FALSE
    
    while(!isLeaf){
      feature <- tree[Node == node, Feature]
      
      # Determine if we're at terminal node. If not, continue.
      if (feature == 'Leaf'){
        isLeaf <- TRUE
      
      } else{
        split <- tree[Node == node, Split]
        xVal <- xVec[[feature]]
        
        # If value is missing, progress to the Missing node
        if (is.na(xVal)){
          node <- tree[Node == node, Missing]
          
        # If value not missing, progress to the Yes/No node
        } else{
          node <- ifelse(xVec[[feature]] < split
                         , tree[Node == node, Yes]
                         , tree[Node == node, No])
        }
      }
    }
    embeddedMat[i, node] <- 1
  }
  
  return(embeddedMat)
}

#' @title Embed data in high-dimensional space with a boosted tree model.
#' @name EmbedBooster
#' @description Embed a dataset into a high-dimensional binary space with a boosted tree model.
#' Largely just assembles trees from XGBoost model and sends data through each tree.
#' @param x input data (matrix, data.frame, or data.table). Shape is n x p.
#' @param model an xgb.Booster model
#' @return data embedded into high-dim space
#' @export
EmbedBooster <- function(x, model){
  
  # Get trees!
  featureNames <- names(x)
  trees <- GetTreeTable(featureNames
                    , model = model)
  treeIds <- unique(trees[, Tree])
  
  matList <- list()
  for(treeId in treeIds){
    tree <- trees[Tree == treeId]
    matList <- c(matList
                 , list(EmbedInTree(x
                                    , tree)))
  }
  
  # columnwise concatenation of individual tree embeddings
  embeddedMat <- do.call(cbind, matList)
  
  # Data quality check. All obs should be sent to one terminal node per tree.
  stopifnot(all(rowSums(embeddedMat) == length(treeIds)))
  
  return(embeddedMat)
}
