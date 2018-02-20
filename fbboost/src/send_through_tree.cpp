#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
int c_intvec_which_equals(IntegerVector x, int val){
  int idx=-1;
  int n = x.size();
  for(int j = 0; j < n; ++j) {
    if(x[j] == val){
      idx = j;
    }
  }
  return idx;
}

// [[Rcpp::export]]
int ThroughTree(List x,
                    Rcpp::IntegerVector Node,
                    Rcpp::CharacterVector Feature,
                    Rcpp::NumericVector Split,
                    Rcpp::IntegerVector Yes,
                    Rcpp::IntegerVector No,
                    Rcpp::IntegerVector Missing) {
  int rowIdx = 0;
  int node = 0;
  bool notLeaf = TRUE;
  double split = NA_REAL;
  double xVal = NA_REAL;
  std::string feat;

  while (notLeaf) {
    // Rcout << "preceding rowIdx is " << rowIdx << std::endl;
    feat = Feature[rowIdx];
    // Rcout << "feat is " << feat << std::endl;

    if (feat == "Leaf"){
      notLeaf = FALSE;
    } else{
      split = Split[rowIdx];
      xVal = x[feat];
      // Rcout << "split is " << split << std::endl;
      // Rcout << "xVal is " << xVal << std::endl;
      
      if (NumericVector::is_na(xVal)){
        // Rcout << "xVal is missing " << std::endl;
        node = Missing[rowIdx];
      } else{
        // Go to row where Node ID is Yes[node]
        if (xVal < split){
          // Rcout << "Go Left" << std::endl;
          node = Yes[rowIdx];
        } else {
          // Rcout << "Go Right" << std::endl;
          node = No[rowIdx];
        }
      }
    }
    // Rcout << "node is " << node << std::endl;
    rowIdx = c_intvec_which_equals(Node, node);
  }
    
  return node;
}


/*** R
# require(data.table)
# irisDT <- as.data.table(iris)
# x <- irisDT[1]
# x[1, 'Petal.Width'] <- NA_real_
# ThroughTree(x
#             , Node = c(0, 1, 2)
#             , Feature = c('Petal.Width', 'Leaf', 'Leaf')
#             , Split = c(2.45, NA_real_, NA_real_)
#             , Yes = c(1, NA_integer_, NA_integer_)
#             , No = c(2, NA_integer_, NA_integer_)
#             , Missing = c(1, NA_integer_, NA_integer_))
*/
