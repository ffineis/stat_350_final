#' @title Back out trees from boosted tree model from xgboost package
#' @name GetTrees
#' @description Make the output of xgboost::xgb.model.dt.tree more useful
#' @param featureNames names of data as it was sent into xgboost::xgb.train
#' @param model an xgb.Booster model
#' @return a data.table of trees
#' @importFrom xgboost xgb.model.dt.tree
#' @importFrom data.table :=
GetTrees <- function(featureNames, model){
  trees <- xgboost::xgb.model.dt.tree(featureNames
                                      , model = model)
  trees[, Tree := Tree + 1]
  trees[, Node := Node + 1]
  
  for(field in c('ID', 'Yes', 'No', 'Missing')){
    trees[, eval(field) := unlist(lapply(get(field)
                                  , FUN = function(x){1 + as.integer(strsplit(x, '-')[[1]][2])}))]
  }

  return(trees)
}

#' @title Embed data in a single tree created from GetTrees
#' @name EmbedInTree
#' @description Send a datapoint through a decision tree, embedding the data in binary tree-leaf space, that tree
#' being a structure created with the GetTrees function.
#' @param x input data (matrix, data.frame, or data.table). Shape is n x p.
#' @param tree processed data.table output from xgboost::xgb.model.dt.tree
#' @return binary embedding matrix. Of shape n x #-terminal-nodes-in-tree
EmbedInTree <- function(x, tree){
  
  nObs <- dim(x)[1]
  embeddedMat <- matrix(0
                        , nrow = nObs
                        , ncol = max(tree[, Node]))
  
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
  trees <- GetTrees(featureNames
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
