#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
int ThroughTree(List x,
                    Rcpp::IntegerVector Node,
                    Rcpp::CharacterVector Feature,
                    Rcpp::NumericVector Split,
                    Rcpp::IntegerVector Yes,
                    Rcpp::IntegerVector No,
                    Rcpp::IntegerVector Missing) {
  int node = 0;
  bool notLeaf = TRUE;
  double split = NA_REAL;
  double xVal = NA_REAL;
  std::string feat;

  feat = Rcpp::as<std::string>(Feature[node]);
  notLeaf = TRUE;

  while (notLeaf) {
    feat = Feature[node];
    
    if (feat == "Leaf"){
      notLeaf = FALSE;
    } else{
      split = Split[node];
      xVal = x[feat];
      
      if (ISNA(xVal)){
        node = Missing[node];
      } else{
        if (xVal < split){
          node = Yes[node] - 1;
        } else {
          node = No[node] - 1;
        }
      }
    }
  }
    
  return node + 1;
}


/*** R
# require(data.table)
# irisDT <- as.data.table(iris)
# x <- irisDT[1]
# x[1, 'Petal.Width'] <- NA_real_
# ThroughTree(x
#             , Node = c(1, 2, 3)
#             , Feature = c('Petal.Width', 'Leaf', 'Leaf')
#             , Split = c(2.45, NA_real_, NA_real_)
#             , Yes = c(2, NA_integer_, NA_integer_)
#             , No = c(3, NA_integer_, NA_integer_)
#             , Missing = c(2, NA_integer_, NA_integer_))
*/
