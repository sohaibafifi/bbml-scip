#pragma once
#include <unordered_map>
#include <vector>
#include "scip/scip.h"

namespace bbml {

struct CandidateFeature {
  int var_index;
  double obj, reduced_cost, fracval, domain_width;
  int is_binary, is_integer, is_indicator, is_sos;
  double pseudocost_up, pseudocost_down;
  int pc_obs_up, pc_obs_down;
  // Additional runtime state
  int at_lb, at_ub;     // whether current LP solution is at local bounds
  int col_nnz;          // number of nonzeros in LP column (approx degree)
};

struct NodeFeature {
  int depth;
  double best_bound, incumbent, gap, lp_time;
  int lp_iterations;
  int cut_rounds, refactor_count;
  double cond_est;
};

struct ExtractedFeatures {
  std::vector<CandidateFeature> candidates;
  NodeFeature node;
  struct GraphData {
    int n_var = 0;
    int n_con = 0;
    int d_var = 0;
    int d_con = 0;
    std::vector<double> var_feat;   // [n_var * d_var]
    std::vector<double> con_feat;   // [n_con * d_con]
    std::vector<int> edge_rows;     // [E]
    std::vector<int> edge_cols;     // [E]
    std::vector<double> edge_vals;  // [E]
  } graph;
};

class FeatureExtractor {
 public:
  ExtractedFeatures fromSCIP(SCIP* scip, SCIP_NODE* node) const;

 private:
  struct StaticVarFeat {
    double obj;
    int is_binary;
    int is_integer;
    int is_indicator;
    int is_sos;
  };
  mutable std::unordered_map<int, StaticVarFeat> static_cache_;
};

}  // namespace bbml
