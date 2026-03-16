#pragma once
#include <string>
#include <utility>
#include <vector>
#include "bbml/feature_extractor.hpp"

namespace bbml {

class OnnxRunner {
 public:
  explicit OnnxRunner(const std::string& model_path);
  ~OnnxRunner();
  std::pair<std::vector<double>, double> score_candidates(
      const ExtractedFeatures& feats);
  bool requires_graph_inputs();

  const std::string& model_path() const { return model_path_; }
  void set_model_path(const std::string& path) { model_path_ = path; }

 private:
  std::string model_path_;
#ifdef BBML_WITH_ONNX
  struct OrtHolder;
  OrtHolder* ort_{nullptr};
  void ensure_session_();
  void destroy_session_();
#endif
};

}  // namespace bbml
