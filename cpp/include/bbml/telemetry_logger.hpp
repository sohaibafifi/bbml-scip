#pragma once
#include <fstream>
#include <mutex>  // NOLINT(build/c++11)
#include <optional>
#include <string>
#include <vector>
#include "bbml/feature_extractor.hpp"
#include "scip/scip.h"

namespace bbml {

class TelemetryLogger {
 public:
  TelemetryLogger();
  ~TelemetryLogger();

  void set_path(const std::string& path);
  void set_instance_id(const std::string& instance_id);
  void set_append(bool append) { append_ = append; }
  void set_graph_path(const std::string& path);
  void set_alpha_path(const std::string& path);

  // Log one line per candidate for a given node
  void log_node_candidates(SCIP* scip,
                           SCIP_NODE* node,
                           const ExtractedFeatures& feats,
                           int chosen_idx,
                           const std::vector<double>* sb_up = nullptr,
                           const std::vector<double>* sb_down = nullptr);

  // Log a single root-level event capturing global metrics
  void log_root(SCIP* scip);

  // Log a graph snapshot per node (var_feat, con_feat, edge_index)
  void log_graph_snapshot(SCIP* scip,
                          SCIP_NODE* node,
                          const ExtractedFeatures& feats,
                          int chosen_idx,
                          const std::vector<double>* sb_up = nullptr,
                          const std::vector<double>* sb_down = nullptr);

  void log_alpha_decision(SCIP_NODE* node,
                          const ExtractedFeatures& feats,
                          double alpha,
                          double confidence,
                          const std::string& fallback_reason);

 private:
  std::mutex mtx_;
  std::ofstream out_;
  std::ofstream out_graph_;
  std::ofstream out_alpha_;
  std::string path_;
  std::string graph_path_;
  std::string alpha_path_;
  std::string instance_id_;
  bool open_{false};
  bool open_graph_{false};
  bool open_alpha_{false};
  bool append_{true};

  void ensure_open_();
  void ensure_open_graph_();
  void ensure_open_alpha_();
};

// Singleton accessors (simple and sufficient for plugin scope)
TelemetryLogger& getTelemetryLogger();
void setTelemetryPath(const std::string& path);
void setTelemetryInstanceId(const std::string& id);
void setTelemetryGraphPath(const std::string& path);
void setTelemetryAlphaPath(const std::string& path);

}  // namespace bbml
