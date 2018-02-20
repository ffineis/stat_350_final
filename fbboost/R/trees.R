#' @title Back out trees from boosted tree model from xgboost package
#' @name GetTreeTable
#' @description Make the output of xgboost::xgb.model.dt.tree more useful
#' @param featureNames names of data as it was sent into xgboost::xgb.train
#' @param model an xgb.Booster model
#' @return a data.table of trees
#' @importFrom xgboost xgb.model.dt.tree
#' @importFrom data.table := rbindlist
#' @export
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
  
  trees[, Split := as.numeric(Split)]

  # change xgboost tree learner representation to something more intuitive
  for(field in c('ID', 'Yes', 'No', 'Missing')){
    trees[, eval(field) := unlist(lapply(get(field)
                                  , FUN = function(x){as.integer(strsplit(x, '-')[[1]][2])}))]
  }
  
  # Remove any trees that may have been duplicated (on the off chance)
  treeList <- lapply(1:max(unique(trees$Tree))
                     , FUN= function(x){trees[Tree == x]})
  treeList <- unique(treeList)
  trees <- rbindlist(treeList)

  return(trees)
}

#' @title Embed data in a single tree created from GetTreeTable
#' @name EmbedInTree
#' @description Send a datapoint through a decision tree, embedding the data in binary tree-leaf space, that tree
#' being a structure created with the GetTreeTable function.
#' @param x input data (matrix, data.frame, or data.table). Shape is n x p.
#' @param tree processed data.table output from xgboost::xgb.model.dt.tree
#' @return binary embedding matrix. Of shape n x #-terminal-nodes-in-tree
#' @useDynLib fbboost
#' @importFrom Rcpp sourceCpp
#' @export
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
  
  # Ensure that the tree is ordered by ascending Node ID.
  tree <- tree[order(Node)]
  
  # output storage
  embeddedMat <- matrix(0
                        , nrow = nObs
                        , ncol = max(tree[, Node]))
  
  # embed all observations in tree space
  for(i in 1:nObs){
    node <- ThroughTree(x[i, ]
                        , Node = tree[, Node]
                        , Feature = tree[, Feature]
                        , Split = tree[, Split]
                        , Yes = tree[, Yes]
                        , No = tree[, No]
                        , Missing = tree[, Missing])
    
    # xVec <- x[i, ]
    # node <- 1
    # isLeaf <- FALSE
    # 
    # while(!isLeaf){
    #   feature <- tree[Node == node, Feature]
    #   
    #   # Determine if we're at terminal node. If not, continue.
    #   if (feature == 'Leaf'){
    #     isLeaf <- TRUE
    #   
    #   } else{
    #     split <- tree[Node == node, Split]
    #     xVal <- xVec[[feature]]
    #     
    #     # If value is missing, progress to the Missing node
    #     if (is.na(xVal)){
    #       node <- tree[Node == node, Missing]
    #       
    #     # If value not missing, progress to the Yes/No node
    #     } else{
    #       node <- ifelse(xVal < split
    #                      , tree[Node == node, Yes]
    #                      , tree[Node == node, No])
    #     }
    #   }
    # }
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
#' @return list of 2: 'data': data embedded into high-dim space, 'treeCuts': vector defining each tree's embedding.
#' @importFrom doMC registerDoMC
#' @importFrom foreach foreach %dopar%
#' @export
EmbedBooster <- function(x, model, nJobs=parallel::detectCores()-1){
  
  # Get trees!
  featureNames <- names(x)
  trees <- GetTreeTable(featureNames
                    , model = model)
  treeIds <- unique(trees[, Tree])
  nTrees <- max(treeIds)

  if(nJobs > 1){    
    # Create parallel cluster
    if (nJobs > parallel::detectCores()){
      stop('nJobs must be <= number of available cores.')
    }
    doMC::registerDoMC(nJobs)

    # distribute embedding through trees over clusters
    mats <- foreach::foreach(i = seq_len(nTrees)) %dopar% {
      mat <- EmbedInTree(x
                         , tree = trees[Tree == i])
      return(mat)
    }
    # don't ask me why foreach returns such a weird array. Need to flatten.
    matList <- list()
    for(i in 1:length(mats)){
      matList[[i]] <- mats[i][[1]]
    }
  } else {
    matList <- list()
    for(treeId in seq_len(nTrees)){
      tree <- trees[Tree == treeId]
      matList <- c(matList
                   , list(EmbedInTree(x
                                      , tree)))
    }
  }
  
  # columnwise concatenation of individual tree embeddings
  embeddedMat <- do.call(cbind, matList)
  treeCuts <- unlist(lapply(matList, FUN = function(x){dim(x)[2]}))
  
  return(list('data' = embeddedMat
              , 'treeCuts' = treeCuts))
}
