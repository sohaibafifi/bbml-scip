#include "bbml/telemetry_logger.hpp"
#include <cstdlib>
#include <filesystem>  // NOLINT(build/c++17)
#include <iomanip>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace bbml {

static TelemetryLogger* g_logger = nullptr;

TelemetryLogger::TelemetryLogger() {
  const char* envpath = std::getenv("BBML_TELEMETRY_PATH");
  const char* envid = std::getenv("BBML_INSTANCE_ID");
  path_ = envpath ? std::string(envpath)
                  : std::string("data/logs/train.ndjson");
  instance_id_ = envid ? std::string(envid) : std::string("unknown");
}

TelemetryLogger::~TelemetryLogger() {
  std::lock_guard<std::mutex> lk(mtx_);
  if (out_.is_open()) {
    out_.close();
  }
  if (out_graph_.is_open()) {
    out_graph_.close();
  }
}

void TelemetryLogger::set_path(const std::string& path) {
  std::lock_guard<std::mutex> lk(mtx_);
  path_ = path;
  if (out_.is_open()) {
    out_.close();
    open_ = false;
  }
}

void TelemetryLogger::set_instance_id(const std::string& instance_id) {
  std::lock_guard<std::mutex> lk(mtx_);
  instance_id_ = instance_id;
}

void TelemetryLogger::ensure_open_() {
  if (open_) {
    return;
  }
  fs::path p(path_);
  if (p.has_parent_path()) {
    std::error_code ec;
    fs::create_directories(p.parent_path(), ec);
    (void)ec;
  }
  std::ios::openmode mode = std::ios::out | (append_ ? std::ios::app : std::ios::trunc);
  out_.open(path_, mode);
  // after first open, default to append mode to avoid truncating on subsequent reopens
  append_ = true;
  open_ = out_.is_open();
}

void TelemetryLogger::set_graph_path(const std::string& path) {
  std::lock_guard<std::mutex> lk(mtx_);
  graph_path_ = path;
  if (out_graph_.is_open()) {
    out_graph_.close();
    open_graph_ = false;
  }
}

void TelemetryLogger::ensure_open_graph_() {
  if (open_graph_) {
    return;
  }
  if (graph_path_.empty()) {
    return;
  }
  fs::path p(graph_path_);
  if (p.has_parent_path()) {
    std::error_code ec;
    fs::create_directories(p.parent_path(), ec);
    (void)ec;
  }
  std::ios::openmode mode = std::ios::out | (append_ ? std::ios::app : std::ios::trunc);
  out_graph_.open(graph_path_, mode);
  append_ = true;
  open_graph_ = out_graph_.is_open();
}

void TelemetryLogger::log_node_candidates(
    SCIP* scip,
    SCIP_NODE* node,
    const ExtractedFeatures& feats,
    int chosen_idx,
    const std::vector<double>* sb_up,
    const std::vector<double>* sb_down) {
  std::lock_guard<std::mutex> lk(mtx_);
  ensure_open_();
  if (!open_) {
    return;
  }

  // Node identifiers
  SCIP_Longint node_num = 0;
  if (node != nullptr) {
    node_num = SCIPnodeGetNumber(node);
  }
  SCIP_NODE* parent = node ? SCIPnodeGetParent(node) : nullptr;
  SCIP_Longint parent_num = parent ? SCIPnodeGetNumber(parent) : -1;

  // Write one JSON line per candidate (single-line NDJSON)
  for (size_t i = 0; i < feats.candidates.size(); ++i) {
    const auto& c = feats.candidates[i];
    out_ << '{'
         << "\"instance_id\":\"" << instance_id_ << "\","
         << "\"node_id\":" << node_num << ","
         << "\"parent_node_id\":" << parent_num << ","
         << "\"depth\":" << feats.node.depth << ","
         << "\"var_id\":" << c.var_index << ","
         << "\"obj\":" << std::setprecision(17) << c.obj << ","
         << "\"reduced_cost\":" << std::setprecision(17) << c.reduced_cost << ","
         << "\"fracval\":" << std::setprecision(17) << c.fracval << ","
         << "\"domain_width\":" << std::setprecision(17) << c.domain_width << ","
         << "\"pseudocost_up\":" << std::setprecision(17) << c.pseudocost_up << ","
         << "\"pseudocost_down\":" << std::setprecision(17) << c.pseudocost_down << ","
         << "\"pc_obs_up\":" << c.pc_obs_up << ","
         << "\"pc_obs_down\":" << c.pc_obs_down << ","
         << "\"at_lb\":" << c.at_lb << ","
         << "\"at_ub\":" << c.at_ub << ","
         << "\"col_nnz\":" << c.col_nnz << ","
         << "\"is_binary\":" << c.is_binary << ","
         << "\"is_integer\":" << c.is_integer << ","
         << "\"is_indicator\":" << c.is_indicator << ","
         << "\"is_sos\":" << c.is_sos << ","
         << "\"best_bound\":" << std::setprecision(17) << feats.node.best_bound << ","
         << "\"incumbent\":" << std::setprecision(17) << feats.node.incumbent << ","
         << "\"gap\":" << std::setprecision(17) << feats.node.gap << ","
         << "\"lp_time\":" << std::setprecision(17) << feats.node.lp_time << ","
         << "\"lp_iters\":" << feats.node.lp_iterations << ","
         << "\"cut_rounds\":" << feats.node.cut_rounds << ","
         << "\"refactor_count\":" << feats.node.refactor_count << ","
         << "\"cond_est\":" << std::setprecision(17) << feats.node.cond_est << ","
         << "\"time_since_incumbent\":0,"
         << (sb_up && i < sb_up->size() ? (std::string("\"sb_score_up\":") + std::to_string((*sb_up)[i]) + ",") : std::string())
         << (sb_down && i < sb_down->size() ? (std::string("\"sb_score_down\":") + std::to_string((*sb_down)[i]) + ",") : std::string())
         << "\"chosen_idx\":" << chosen_idx << "}" << '\n';
  }
  out_.flush();
}

void TelemetryLogger::log_graph_snapshot(
    SCIP* /*scip*/,
    SCIP_NODE* node,
    const ExtractedFeatures& feats,
    int chosen_idx,
    const std::vector<double>* sb_up,
    const std::vector<double>* sb_down) {
  std::lock_guard<std::mutex> lk(mtx_);
  ensure_open_graph_();
  if (!open_graph_) {
    return;
  }
  // Node identifiers
  SCIP_Longint node_num = 0;
  if (node != nullptr) node_num = SCIPnodeGetNumber(node);
  SCIP_NODE* parent = node ? SCIPnodeGetParent(node) : nullptr;
  SCIP_Longint parent_num = parent ? SCIPnodeGetNumber(parent) : -1;

  out_graph_ << '{'
             << "\"instance_id\":\"" << instance_id_ << "\","
             << "\"node_id\":" << node_num << ","
             << "\"parent_node_id\":" << parent_num << ","
             << "\"depth\":" << feats.node.depth << ",";
  // var_feat
  out_graph_ << "\"var_feat\":[";
  int dvar = feats.graph.d_var;
  for (int i = 0; i < feats.graph.n_var; ++i) {
    if (i) out_graph_ << ',';
    out_graph_ << '[';
    for (int j = 0; j < dvar; ++j) {
      if (j) out_graph_ << ',';
      out_graph_ << std::setprecision(17)
                 << feats.graph.var_feat[static_cast<size_t>(i) * dvar + j];
    }
    out_graph_ << ']';
  }
  out_graph_ << "],";
  // con_feat
  out_graph_ << "\"con_feat\":[";
  int dcon = feats.graph.d_con;
  for (int i = 0; i < feats.graph.n_con; ++i) {
    if (i) out_graph_ << ',';
    out_graph_ << '[';
    for (int j = 0; j < dcon; ++j) {
      if (j) out_graph_ << ',';
      out_graph_ << std::setprecision(17)
                 << feats.graph.con_feat[static_cast<size_t>(i) * dcon + j];
    }
    out_graph_ << ']';
  }
  out_graph_ << "],";
  // edge_index
  out_graph_ << "\"edge_index\":[";
  out_graph_ << '[';
  for (size_t e = 0; e < feats.graph.edge_rows.size(); ++e) {
    if (e) out_graph_ << ',';
    out_graph_ << feats.graph.edge_rows[e];
  }
  out_graph_ << "],";
  out_graph_ << '[';
  for (size_t e = 0; e < feats.graph.edge_cols.size(); ++e) {
    if (e) out_graph_ << ',';
    out_graph_ << feats.graph.edge_cols[e];
  }
  out_graph_ << "]],";
  // edge_val
  out_graph_ << "\"edge_val\":[";
  for (size_t e = 0; e < feats.graph.edge_vals.size(); ++e) {
    if (e) out_graph_ << ',';
    out_graph_ << std::setprecision(17) << feats.graph.edge_vals[e];
  }
  out_graph_ << "],";
  // Optional SB
  if (sb_up) {
    out_graph_ << "\"sb_score_up\":[";
    for (size_t i = 0; i < sb_up->size(); ++i) {
      if (i) out_graph_ << ',';
      out_graph_ << std::setprecision(17) << (*sb_up)[i];
    }
    out_graph_ << "],";
  }
  if (sb_down) {
    out_graph_ << "\"sb_score_down\":[";
    for (size_t i = 0; i < sb_down->size(); ++i) {
      if (i) out_graph_ << ',';
      out_graph_ << std::setprecision(17) << (*sb_down)[i];
    }
    out_graph_ << "],";
  }
  out_graph_ << "\"chosen_idx\":" << chosen_idx << "}" << '\n';
  out_graph_.flush();
}

void TelemetryLogger::log_root(SCIP* scip) {
  std::lock_guard<std::mutex> lk(mtx_);
  ensure_open_();
  if (!open_) {
    return;
  }
  double best = SCIPgetDualbound(scip);
  double inc = SCIPgetPrimalbound(scip);
  double gap = SCIPgetGap(scip);
  double lptime = SCIPgetSolvingTime(scip) - SCIPgetReadingTime(scip);
  int lpiter = static_cast<int>(SCIPgetNLPIterations(scip));
  out_ << '{'
       << "\"instance_id\":\"" << instance_id_ << "\","
       << "\"node_id\":0,"
       << "\"parent_node_id\":-1,"
       << "\"depth\":0,"
       << "\"best_bound\":" << std::setprecision(17) << best << ","
       << "\"incumbent\":" << std::setprecision(17) << inc << ","
       << "\"gap\":" << std::setprecision(17) << gap << ","
       << "\"lp_time\":" << std::setprecision(17) << lptime << ","
       << "\"lp_iters\":" << lpiter << "}" << '\n';
  out_.flush();
}

TelemetryLogger& getTelemetryLogger() {
  if (!g_logger) {
    g_logger = new TelemetryLogger();
  }
  return *g_logger;
}

void setTelemetryPath(const std::string& path) {
  getTelemetryLogger().set_path(path);
}

void setTelemetryInstanceId(const std::string& id) {
  getTelemetryLogger().set_instance_id(id);
}

void setTelemetryGraphPath(const std::string& path) {
  getTelemetryLogger().set_graph_path(path);
}

}  // namespace bbml
