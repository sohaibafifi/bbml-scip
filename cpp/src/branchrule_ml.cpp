#include "bbml/branchrule_ml.hpp"
#include <algorithm>
#include <cmath>
#include <memory>
#include <climits>
#include <string>
#include <utility>
#include <vector>
#include "bbml/telemetry_logger.hpp"
#include "scip/scip.h"
#include "scip/scip_var.h"

namespace bbml {

namespace {

double sigmoid(double x) {
  if (x >= 0.0) {
    const double z = std::exp(-x);
    return 1.0 / (1.0 + z);
  }
  const double z = std::exp(x);
  return z / (1.0 + z);
}

}  // namespace

BranchruleML::BranchruleML(std::unique_ptr<OnnxRunner> runner)
    : runner_(std::move(runner)) {}

int BranchruleML::choose(const ExtractedFeatures& feats,
                         const std::vector<double>& fallback_scores,
                         std::vector<double>* blended,
                         double* alpha_used,
                         double* confidence_used,
                         bool* used_runtime_confidence,
                         double confidence,
                         int depth) const {
  auto sc = runner_->score_candidates(feats);
  const auto& ml = sc.first;
  const bool has_runtime_confidence = std::isfinite(sc.second);
  const double effective_confidence = std::clamp(
      has_runtime_confidence ? sc.second : confidence, 0.0, 1.0);
  const double alpha_depth = std::clamp(
      amax_ - depth_penalty_ * depth, amin_, amax_);
  const double confidence_gate = use_confidence_gate_
                                     ? sigmoid(12.0 * (effective_confidence - theta_))
                                     : 1.0;
  const double alpha = amin_ + (alpha_depth - amin_) * confidence_gate;
  if (alpha_used) {
    *alpha_used = alpha;
  }
  if (confidence_used) {
    *confidence_used = effective_confidence;
  }
  if (used_runtime_confidence) {
    *used_runtime_confidence = has_runtime_confidence;
  }
  blended->resize(feats.candidates.size());
  for (size_t i = 0; i < blended->size(); ++i) {
    const double fallback = i < fallback_scores.size() ? fallback_scores[i] : 0.0;
    double mli = (i < ml.size() ? alpha * (ml[i] / std::max(1e-6, temperature_)) : 0.0);
    (*blended)[i] = mli + (1.0 - alpha) * fallback;
  }
  auto it = std::max_element(blended->begin(), blended->end());
  return static_cast<int>(std::distance(blended->begin(), it));
}

namespace {

struct BranchruleData {
  BranchruleML* br = nullptr;
  FeatureExtractor fx;
  // runtime state for hot-reload
  std::string last_model_path;
  int last_reload_counter = 0;
  std::string last_tpath;
};

static bool isCandidateBranchable(SCIP* scip, SCIP_VAR* var) {
  if (var == nullptr) {
    return false;
  }
  SCIP_Real lb = SCIPvarGetLbLocal(var);
  SCIP_Real ub = SCIPvarGetUbLocal(var);
  if (SCIPisGE(scip, lb, ub)) {
    return false;
  }
  SCIP_Real lpsol = SCIPvarGetLPSol(var);
  return !SCIPisFeasIntegral(scip, lpsol);
}

static SCIP_RETCODE BranchruleExecLP(SCIP* scip,
                                     SCIP_BRANCHRULE* branchrule,
                                     SCIP_Bool /*allowaddcons*/,
                                     SCIP_RESULT* result) {
  auto* data = reinterpret_cast<BranchruleData*>(
      SCIPbranchruleGetData(branchrule));
  if (!data || !data->br) {
    *result = SCIP_DIDNOTRUN;
    return SCIP_OKAY;
  }
  SCIP_NODE* focus = SCIPgetFocusNode(scip);
  auto feats = data->fx.fromSCIP(scip, focus);
  std::vector<double> blended;
  double alpha = 0.0;
  // runtime parameters
  SCIP_Bool enable = TRUE;
  SCIP_Bool telemetry = TRUE;
  SCIP_Bool tlogsb = FALSE;
  SCIP_Bool tappend = TRUE;
  SCIP_Bool tgraph = FALSE;
  SCIP_Bool talpha = FALSE;
  SCIP_Bool use_confidence_gate = TRUE;
  SCIP_Real a_min = 0.1, a_max = 0.8, depth_pen = 0.02;
  SCIP_Real a_theta = 0.5;
  SCIP_Real confidence = 0.5;  // base confidence
  SCIP_Real temperature = 1.0;
  SCIP_Real cond_thresh = 1e8;
  char* model_path_c = nullptr;
  int reload_counter = 0;
  char* tpath_c = nullptr;
  char* tgraph_path_c = nullptr;
  char* talpha_path_c = nullptr;
  (void)SCIPgetBoolParam(scip, "bbml/enable", &enable);
  (void)SCIPgetBoolParam(scip, "bbml/telemetry", &telemetry);
  (void)SCIPgetBoolParam(scip, "bbml/telemetry/strongbranch", &tlogsb);
  (void)SCIPgetBoolParam(scip, "bbml/telemetry/graph", &tgraph);
  (void)SCIPgetBoolParam(scip, "bbml/telemetry/alpha", &talpha);
  (void)SCIPgetBoolParam(scip, "bbml/alpha/use_confidence_gate", &use_confidence_gate);
  (void)SCIPgetRealParam(scip, "bbml/alpha/min", &a_min);
  (void)SCIPgetRealParam(scip, "bbml/alpha/max", &a_max);
  (void)SCIPgetRealParam(scip, "bbml/alpha/depth_penalty", &depth_pen);
  (void)SCIPgetRealParam(scip, "bbml/alpha/theta", &a_theta);
  (void)SCIPgetRealParam(scip, "bbml/confidence", &confidence);
  (void)SCIPgetRealParam(scip, "bbml/temperature", &temperature);
  (void)SCIPgetRealParam(scip, "bbml/numerics/cond_threshold", &cond_thresh);
  (void)SCIPgetStringParam(scip, "bbml/model_path", &model_path_c);
  (void)SCIPgetIntParam(scip, "bbml/reload", &reload_counter);
  (void)SCIPgetStringParam(scip, "bbml/telemetry/path", &tpath_c);
  (void)SCIPgetStringParam(scip, "bbml/telemetry/graph_path", &tgraph_path_c);
  (void)SCIPgetStringParam(scip, "bbml/telemetry/alpha_path", &talpha_path_c);
  (void)SCIPgetBoolParam(scip, "bbml/telemetry/append", &tappend);

  if (!enable) {
    *result = SCIP_DIDNOTRUN;
    return SCIP_OKAY;
  }

  // hot-reload model if requested
  std::string model_path = model_path_c ? std::string(model_path_c) : std::string("");
  if (reload_counter != data->last_reload_counter || model_path != data->last_model_path) {
    data->br->set_runner(std::make_unique<OnnxRunner>(model_path));
    data->last_model_path = model_path;
    data->last_reload_counter = reload_counter;
  }

  // update telemetry path if provided
  if (tpath_c != nullptr && tpath_c[0] != '\0') {
    std::string tpath(tpath_c);
    if (tpath != data->last_tpath) {
      getTelemetryLogger().set_path(tpath);
      getTelemetryLogger().set_append(tappend == TRUE);
      data->last_tpath = tpath;
    }
  }
  if (tgraph_path_c != nullptr && tgraph_path_c[0] != '\0') {
    setTelemetryGraphPath(std::string(tgraph_path_c));
  }
  if (talpha_path_c != nullptr && talpha_path_c[0] != '\0') {
    setTelemetryAlphaPath(std::string(talpha_path_c));
  }

  // update blending params
  data->br->set_alpha_params(a_min, a_max, depth_pen);
  data->br->set_alpha_theta(static_cast<double>(a_theta));
  data->br->set_use_confidence_gate(use_confidence_gate == TRUE);
  data->br->set_temperature(static_cast<double>(temperature));
  // gate confidence by numerics (e.g., ill-conditioned LP)
  SCIP_Real eff_conf = (feats.node.cond_est > cond_thresh) ? 0.0 : confidence;
  SCIP_VAR** cands = nullptr;
  SCIP_Real* candssol = nullptr;
  SCIP_Real* candsfrac = nullptr;
  int nc = 0, np = 0, ni = 0;
  SCIPgetLPBranchCands(scip, &cands, &candssol, &candsfrac, &nc, &np, &ni);
  if (nc <= 0) {
    *result = SCIP_DIDNOTRUN;
    return SCIP_OKAY;
  }
  std::vector<double> fallback_scores(static_cast<size_t>(nc), 0.0);
  std::vector<SCIP_VAR*> candvars(static_cast<size_t>(nc), nullptr);
  for (int i = 0; i < nc; ++i) {
    candvars[static_cast<size_t>(i)] = cands[i];
    double score = SCIPgetVarPseudocostScore(scip, cands[i], candssol[i]);
    if ((!std::isfinite(score) || score < 0.0) && static_cast<size_t>(i) < feats.candidates.size()) {
      score = std::fabs(feats.candidates[static_cast<size_t>(i)].reduced_cost);
    }
    fallback_scores[static_cast<size_t>(i)] = score;
  }
  double used_confidence = 0.0;
  bool used_runtime_confidence = false;
  int idx = data->br->choose(
      feats,
      fallback_scores,
      &blended,
      &alpha,
      &used_confidence,
      &used_runtime_confidence,
      eff_conf,
      feats.node.depth);
  if (idx < 0 || idx >= nc) {
    *result = SCIP_DIDNOTRUN;
    return SCIP_OKAY;
  }
  // optional strong-branch scores for telemetry
  std::vector<double> sb_up, sb_down;
  if (telemetry && tlogsb) {
    sb_up.resize(nc, 0.0);
    sb_down.resize(nc, 0.0);
    std::vector<SCIP_Real> downvals(static_cast<size_t>(nc), 0.0);
    std::vector<SCIP_Real> upvals(static_cast<size_t>(nc), 0.0);
    std::vector<SCIP_Bool> downvalid(static_cast<size_t>(nc), FALSE);
    std::vector<SCIP_Bool> upvalid(static_cast<size_t>(nc), FALSE);
    std::vector<SCIP_Bool> downinf(static_cast<size_t>(nc), FALSE);
    std::vector<SCIP_Bool> upinf(static_cast<size_t>(nc), FALSE);
    std::vector<SCIP_Bool> downconflict(static_cast<size_t>(nc), FALSE);
    std::vector<SCIP_Bool> upconflict(static_cast<size_t>(nc), FALSE);
    SCIP_Bool lperr = FALSE;
    SCIP_RETCODE sbrc = SCIPgetVarsStrongbranchesFrac(
        scip,
        cands,
        nc,
        -1,
        downvals.data(),
        upvals.data(),
        downvalid.data(),
        upvalid.data(),
        downinf.data(),
        upinf.data(),
        downconflict.data(),
        upconflict.data(),
        &lperr);
    if (sbrc == SCIP_OKAY && !lperr) {
      for (int i = 0; i < nc; ++i) {
        sb_down[static_cast<size_t>(i)] = static_cast<double>(downvals[static_cast<size_t>(i)]);
        sb_up[static_cast<size_t>(i)] = static_cast<double>(upvals[static_cast<size_t>(i)]);
      }
    }
  }
  // telemetry: log candidate set for this node
  if (telemetry) {
    const std::vector<double>* sup = tlogsb ? &sb_up : nullptr;
    const std::vector<double>* sdown = tlogsb ? &sb_down : nullptr;
    getTelemetryLogger().log_node_candidates(scip, focus, feats, idx, sup, sdown);
    if (tgraph) {
      getTelemetryLogger().log_graph_snapshot(scip, focus, feats, idx, sup, sdown);
    }
  }
  if (telemetry && talpha) {
    const std::string fallback_reason =
        (feats.node.cond_est > cond_thresh)
            ? "numerics"
            : (used_runtime_confidence ? "ensemble" : "static");
    getTelemetryLogger().log_alpha_decision(
        focus,
        feats,
        alpha,
        used_confidence,
        fallback_reason);
  }
  if (!isCandidateBranchable(scip, candvars[static_cast<size_t>(idx)])) {
    int fallback_idx = -1;
    for (int i = 0; i < nc; ++i) {
      if (!isCandidateBranchable(scip, candvars[static_cast<size_t>(i)])) {
        continue;
      }
      if (fallback_idx < 0 || blended[static_cast<size_t>(i)] > blended[static_cast<size_t>(fallback_idx)]) {
        fallback_idx = i;
      }
    }
    if (fallback_idx < 0) {
      *result = SCIP_DIDNOTRUN;
      return SCIP_OKAY;
    }
    idx = fallback_idx;
  }
  SCIP_VAR* var = candvars[static_cast<size_t>(idx)];
  SCIP_Real brpt = SCIPvarGetLPSol(var);
  SCIP_NODE* down = nullptr;
  SCIP_NODE* eq = nullptr;
  SCIP_NODE* up = nullptr;
  SCIP_CALL(SCIPbranchVarVal(scip, var, brpt, &down, &eq, &up));
  *result = SCIP_BRANCHED;
  return SCIP_OKAY;
}

}  // namespace

SCIP_RETCODE includeBranchruleML(SCIP* scip) {
  SCIP_BRANCHRULE* br = nullptr;
  auto* data = new BranchruleData();
  auto runner = std::make_unique<OnnxRunner>(std::string(""));
  auto* ml = new BranchruleML(std::move(runner));
  data->br = ml;
  SCIP_CALL(SCIPincludeBranchruleBasic(
      scip, &br, "bbml_branch", "ML-assisted branching (proxy)",
      100000, -1, 1.0, reinterpret_cast<SCIP_BRANCHRULEDATA*>(data)));
  SCIP_CALL(SCIPsetBranchruleExecLp(scip, br, BranchruleExecLP));
  // Runtime parameters (global)
  SCIP_CALL(SCIPaddBoolParam(scip, "bbml/enable",
      "enable ML-assisted branching", /*valueptr*/ nullptr, /*isadvanced*/ FALSE,
      /*default*/ TRUE, /*paramchgd*/ nullptr, /*paramdata*/ nullptr));
  SCIP_CALL(SCIPaddBoolParam(scip, "bbml/telemetry",
      "enable telemetry logging from branchrule", /*valueptr*/ nullptr, /*isadvanced*/ FALSE,
      /*default*/ TRUE, /*paramchgd*/ nullptr, /*paramdata*/ nullptr));
  SCIP_CALL(SCIPaddBoolParam(scip, "bbml/telemetry/strongbranch",
      "log strong-branching scores in telemetry (expensive)", /*valueptr*/ nullptr, /*isadvanced*/ TRUE,
      /*default*/ FALSE, /*paramchgd*/ nullptr, /*paramdata*/ nullptr));
  SCIP_CALL(SCIPaddBoolParam(scip, "bbml/telemetry/graph",
      "log graph snapshots (var, con, edges) per node", /*valueptr*/ nullptr, /*isadvanced*/ TRUE,
      /*default*/ FALSE, /*paramchgd*/ nullptr, /*paramdata*/ nullptr));
  SCIP_CALL(SCIPaddStringParam(scip, "bbml/model_path",
      "ONNX model path for scoring", /*valueptr*/ nullptr, /*isadvanced*/ FALSE,
      /*default*/ "", /*paramchgd*/ nullptr, /*paramdata*/ nullptr));
  SCIP_CALL(SCIPaddIntParam(scip, "bbml/reload",
      "increment to hot-reload model", /*valueptr*/ nullptr, /*isadvanced*/ FALSE,
      /*default*/ 0, /*min*/ 0, /*max*/ INT_MAX, /*paramchgd*/ nullptr, /*paramdata*/ nullptr));
  SCIP_CALL(SCIPaddRealParam(scip, "bbml/alpha/min",
      "minimum alpha for blending", /*valueptr*/ nullptr, /*isadvanced*/ FALSE,
      /*default*/ 0.1, /*min*/ 0.0, /*max*/ 1.0, /*paramchgd*/ nullptr, /*paramdata*/ nullptr));
  SCIP_CALL(SCIPaddRealParam(scip, "bbml/alpha/max",
      "maximum alpha for blending", /*valueptr*/ nullptr, /*isadvanced*/ FALSE,
      /*default*/ 0.8, /*min*/ 0.0, /*max*/ 1.0, /*paramchgd*/ nullptr, /*paramdata*/ nullptr));
  SCIP_CALL(SCIPaddRealParam(scip, "bbml/alpha/depth_penalty",
      "alpha depth penalty per level", /*valueptr*/ nullptr, /*isadvanced*/ FALSE,
      /*default*/ 0.02, /*min*/ 0.0, /*max*/ 1.0, /*paramchgd*/ nullptr, /*paramdata*/ nullptr));
  SCIP_CALL(SCIPaddRealParam(scip, "bbml/alpha/theta",
      "confidence midpoint for alpha gating", /*valueptr*/ nullptr, /*isadvanced*/ FALSE,
      /*default*/ 0.5, /*min*/ 0.0, /*max*/ 1.0, /*paramchgd*/ nullptr, /*paramdata*/ nullptr));
  SCIP_CALL(SCIPaddBoolParam(scip, "bbml/alpha/use_confidence_gate",
      "whether alpha uses the runtime confidence sigmoid gate", /*valueptr*/ nullptr, /*isadvanced*/ FALSE,
      /*default*/ TRUE, /*paramchgd*/ nullptr, /*paramdata*/ nullptr));
  SCIP_CALL(SCIPaddRealParam(scip, "bbml/confidence",
      "fallback confidence used when runtime uncertainty is unavailable", /*valueptr*/ nullptr, /*isadvanced*/ FALSE,
      /*default*/ 0.5, /*min*/ 0.0, /*max*/ 1.0, /*paramchgd*/ nullptr, /*paramdata*/ nullptr));
  SCIP_CALL(SCIPaddRealParam(scip, "bbml/temperature",
      "temperature to scale ML scores (>0)", /*valueptr*/ nullptr, /*isadvanced*/ FALSE,
      /*default*/ 1.0, /*min*/ 1e-6, /*max*/ 1e6, /*paramchgd*/ nullptr, /*paramdata*/ nullptr));
  SCIP_CALL(SCIPaddRealParam(scip, "bbml/numerics/cond_threshold",
      "condition number threshold to gate ML (set alpha=0)", /*valueptr*/ nullptr,
      /*isadvanced*/ FALSE, /*default*/ 1e8, /*min*/ 0.0, /*max*/ SCIP_REAL_MAX,
      /*paramchgd*/ nullptr, /*paramdata*/ nullptr));
  SCIP_CALL(SCIPaddStringParam(scip, "bbml/telemetry/path",
      "telemetry output path (NDJSON)", /*valueptr*/ nullptr, /*isadvanced*/ FALSE,
      /*default*/ "", /*paramchgd*/ nullptr, /*paramdata*/ nullptr));
  SCIP_CALL(SCIPaddStringParam(scip, "bbml/telemetry/graph_path",
      "graph telemetry output path (NDJSON)", /*valueptr*/ nullptr, /*isadvanced*/ TRUE,
      /*default*/ "", /*paramchgd*/ nullptr, /*paramdata*/ nullptr));
  SCIP_CALL(SCIPaddBoolParam(scip, "bbml/telemetry/alpha",
      "log alpha/confidence decisions (CSV)", /*valueptr*/ nullptr, /*isadvanced*/ TRUE,
      /*default*/ FALSE, /*paramchgd*/ nullptr, /*paramdata*/ nullptr));
  SCIP_CALL(SCIPaddStringParam(scip, "bbml/telemetry/alpha_path",
      "alpha telemetry output path (CSV)", /*valueptr*/ nullptr, /*isadvanced*/ TRUE,
      /*default*/ "", /*paramchgd*/ nullptr, /*paramdata*/ nullptr));
  SCIP_CALL(SCIPaddBoolParam(scip, "bbml/telemetry/append",
      "append to telemetry file instead of truncating on first open", /*valueptr*/ nullptr,
      /*isadvanced*/ FALSE, /*default*/ TRUE, /*paramchgd*/ nullptr, /*paramdata*/ nullptr));
  return SCIP_OKAY;
}

}  // namespace bbml
