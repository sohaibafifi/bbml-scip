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
             std::vector<double>* blended_scores,
             double* alpha_used,
             double confidence,
             int depth) const;

  void set_runner(std::unique_ptr<OnnxRunner> runner) { runner_ = std::move(runner); }
  void set_alpha_params(double amin, double amax, double depth_penalty) {
    amin_ = amin; amax_ = amax; depth_penalty_ = depth_penalty;
  }
  void set_temperature(double T) { temperature_ = (T <= 0.0 ? 1.0 : T); }

 private:
  std::unique_ptr<OnnxRunner> runner_;
  double amin_ = 0.1;
  double amax_ = 0.8;
  double depth_penalty_ = 0.02;
  double temperature_ = 1.0;
};

SCIP_RETCODE includeBranchruleML(SCIP* scip);

}  // namespace bbml
