#pragma once
#include <memory>
#include <vector>
#include <utility>
#include "bbml/feature_extractor.hpp"
#include "bbml/onnx_runner.hpp"
#include "scip/scip.h"

namespace bbml {

class BranchruleML {
 public:
  explicit BranchruleML(std::unique_ptr<OnnxRunner> runner);
  int choose(const ExtractedFeatures& feats,
             const std::vector<double>& fallback_scores,
             std::vector<double>* blended_scores,
             double* alpha_used,
             double* confidence_used,
             bool* used_runtime_confidence,
             double confidence,
             int depth) const;

  void set_runner(std::unique_ptr<OnnxRunner> runner) { runner_ = std::move(runner); }
  void set_alpha_params(double amin, double amax, double depth_penalty) {
    amin_ = amin; amax_ = amax; depth_penalty_ = depth_penalty;
  }
  void set_alpha_theta(double theta) { theta_ = theta; }
  void set_use_confidence_gate(bool use_confidence_gate) {
    use_confidence_gate_ = use_confidence_gate;
  }
  void set_temperature(double T) { temperature_ = (T <= 0.0 ? 1.0 : T); }
  bool requires_graph_inputs() const { return runner_ != nullptr && runner_->requires_graph_inputs(); }

 private:
  std::unique_ptr<OnnxRunner> runner_;
  double amin_ = 0.1;
  double amax_ = 0.8;
  double depth_penalty_ = 0.02;
  double theta_ = 0.5;
  double temperature_ = 1.0;
  bool use_confidence_gate_ = true;
};

SCIP_RETCODE includeBranchruleML(SCIP* scip);

}  // namespace bbml
