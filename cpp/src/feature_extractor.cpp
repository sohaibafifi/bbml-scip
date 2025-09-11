#include "bbml/feature_extractor.hpp"
#include <cmath>
#include <unordered_map>
#include <vector>
#include "scip/lp.h"
namespace bbml {

ExtractedFeatures FeatureExtractor::fromSCIP(SCIP* scip,
                                             SCIP_NODE* node) const {
  ExtractedFeatures out{};
  out.node.depth = (node != nullptr) ? SCIPnodeGetDepth(node) : 0;
  out.node.best_bound = SCIPgetDualbound(scip);
  out.node.incumbent = SCIPgetPrimalbound(scip);
  out.node.gap = SCIPgetGap(scip);
  out.node.lp_time = SCIPgetSolvingTime(scip) - SCIPgetReadingTime(scip);
  out.node.lp_iterations = static_cast<int>(SCIPgetNLPIterations(scip));
  out.node.cut_rounds = 0;
  out.node.refactor_count = 0;
  out.node.cond_est = 0.0;  // leave unset unless a stable API is present (guarded below)
  SCIP_VAR **cands = nullptr;
  SCIP_Real *candssol = nullptr;
  SCIP_Real *candsfrac = nullptr;
  int ncands = 0, nprio = 0, nimpl = 0;
  // Only query LP branch candidates when in solving and a focus node exists
  if (SCIPgetFocusNode(scip) != nullptr) {
    SCIPgetLPBranchCands(scip,
                         &cands,
                         &candssol,
                         &candsfrac,
                         &ncands,
                         &nprio,
                         &nimpl);
  }
  out.candidates.reserve(ncands);
  std::unordered_map<int, int> cand_index; cand_index.reserve(ncands);
  for (int i = 0; i < ncands; ++i) {
    SCIP_VAR *v = cands[i];
    CandidateFeature cf{};
    cf.var_index = SCIPvarGetProbindex(v);
    // Static features via cache
    auto it = static_cache_.find(cf.var_index);
    if (it == static_cache_.end()) {
      StaticVarFeat sv{};
      sv.obj = SCIPvarGetObj(v);
      auto t = SCIPvarGetType(v);
      sv.is_binary = (t == SCIP_VARTYPE_BINARY);
      sv.is_integer = (t == SCIP_VARTYPE_INTEGER || t == SCIP_VARTYPE_IMPLINT);
      sv.is_indicator = 0;  // TODO(lafifi): fill from model data if available
      sv.is_sos = 0;        // TODO(lafifi): fill from model data if available
      it = static_cache_.emplace(cf.var_index, sv).first;
    }
    cf.obj = it->second.obj;
    cf.reduced_cost = SCIPgetVarRedcost(scip, v);
    double x = SCIPvarGetLPSol(v);
    cf.fracval = x - std::floor(x);
    cf.domain_width = SCIPvarGetUbLocal(v) - SCIPvarGetLbLocal(v);
    cf.is_binary = it->second.is_binary;
    cf.is_integer = it->second.is_integer;
    cf.is_indicator = it->second.is_indicator;
    cf.is_sos = it->second.is_sos;
    cf.pseudocost_up = 0.0;
    cf.pseudocost_down = 0.0;
    cf.pc_obs_up = 0;
    cf.pc_obs_down = 0;
    // Bounds status at current LP solution
    SCIP_Real lb = SCIPvarGetLbLocal(v);
    SCIP_Real ub = SCIPvarGetUbLocal(v);
    cf.at_lb = (SCIPisEQ(scip, x, lb) ? 1 : 0);
    cf.at_ub = (SCIPisEQ(scip, x, ub) ? 1 : 0);
    // LP column nonzeros (if available)
    cf.col_nnz = 0;
    SCIP_COL* col = SCIPvarGetCol(v);
    if (col != nullptr) {
      cf.col_nnz = static_cast<int>(col->len);
    }
    out.candidates.push_back(cf);
    cand_index.emplace(cf.var_index, i);
  }

  // Build graph snapshot (constraints ↔ candidate variables)
  // var_feat: [ncands, d_var]
  const int dvar = 9;  // obj, redcost, frac, domw, is_bin, is_int, at_lb, at_ub, col_nnz
  out.graph.n_var = ncands;
  out.graph.d_var = dvar;
  out.graph.var_feat.resize(static_cast<size_t>(ncands) * dvar);
  for (int i = 0; i < ncands; ++i) {
    const auto& c = out.candidates[static_cast<size_t>(i)];
    double* dst = &out.graph.var_feat[static_cast<size_t>(i) * dvar];
    dst[0] = c.obj;
    dst[1] = c.reduced_cost;
    dst[2] = c.fracval;
    dst[3] = c.domain_width;
    dst[4] = static_cast<double>(c.is_binary);
    dst[5] = static_cast<double>(c.is_integer);
    dst[6] = static_cast<double>(c.at_lb);
    dst[7] = static_cast<double>(c.at_ub);
    dst[8] = static_cast<double>(c.col_nnz);
  }

  // Build constraint features and edges by scanning candidate columns; avoids SCIPgetLP
  struct RowFeatRec { int nnz; double l1; double lhs; double rhs; };
  std::unordered_map<const SCIP_ROW*, int> row_index;
  std::vector<RowFeatRec> row_feats;
  for (int i = 0; i < ncands; ++i) {
    SCIP_VAR* v = cands[i];
    SCIP_COL* col = SCIPvarGetCol(v);
    if (col == nullptr) continue;
    int nnz = SCIPcolGetNNonz(col);
    SCIP_ROW** rows = SCIPcolGetRows(col);
    SCIP_Real* vals = SCIPcolGetVals(col);
    for (int k = 0; k < nnz; ++k) {
      SCIP_ROW* row = rows[k];
      int ridx;
      auto it = row_index.find(row);
      if (it == row_index.end()) {
        ridx = static_cast<int>(row_index.size());
        row_index.emplace(row, ridx);
        // Compute row features once
        int rnnz = SCIProwGetNNonz(row);
        SCIP_Real* rvals = SCIProwGetVals(row);
        double l1 = 0.0;
        for (int t = 0; t < rnnz; ++t) l1 += std::fabs(static_cast<double>(rvals[t]));
        RowFeatRec rec{rnnz, l1,
                       static_cast<double>(SCIProwGetLhs(row)),
                       static_cast<double>(SCIProwGetRhs(row))};
        row_feats.push_back(rec);
      } else {
        ridx = it->second;
      }
      // Edge from constraint row → candidate variable i
      out.graph.edge_rows.push_back(ridx);
      out.graph.edge_cols.push_back(i);
      out.graph.edge_vals.push_back(static_cast<double>(vals[k]));
    }
  }
  // Finalize constraint feature matrix
  out.graph.n_con = static_cast<int>(row_index.size());
  const int dcon = 4;  // nnz, l1norm, lhs, rhs
  out.graph.d_con = dcon;
  out.graph.con_feat.resize(static_cast<size_t>(out.graph.n_con) * dcon);
  for (int r = 0; r < out.graph.n_con; ++r) {
    const auto& rf = row_feats[static_cast<size_t>(r)];
    double* dst = &out.graph.con_feat[static_cast<size_t>(r) * dcon];
    dst[0] = static_cast<double>(rf.nnz);
    dst[1] = rf.l1;
    dst[2] = rf.lhs;
    dst[3] = rf.rhs;
  }
  return out;
}

}  // namespace bbml
