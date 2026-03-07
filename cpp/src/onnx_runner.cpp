#include "bbml/onnx_runner.hpp"
#include <string>
#include <utility>
#include <vector>
#include <memory>
#include <algorithm>

#ifdef BBML_WITH_ONNX
#include <array>
#include <cstring>
#include "onnxruntime_cxx_api.h"
#endif

namespace bbml {

#ifdef BBML_WITH_ONNX
struct OnnxRunner::OrtHolder {
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "bbml"};
  Ort::SessionOptions opts;
  std::unique_ptr<Ort::Session> session;
  std::vector<std::string> input_name_strs;
  std::vector<std::string> output_name_strs;
  size_t input_dim = 0;  // used for 1-input MLP signature
  size_t input_count = 0;
};

void OnnxRunner::ensure_session_() {
  if (ort_ && ort_->session) return;
  if (!ort_) ort_ = new OrtHolder();
  ort_->opts.SetIntraOpNumThreads(1);
  ort_->opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);
  ort_->session.reset(new Ort::Session(ort_->env, model_path_.c_str(), ort_->opts));
  ort_->input_name_strs.clear();
  ort_->output_name_strs.clear();
  Ort::AllocatorWithDefaultOptions alloc;
  size_t icount = ort_->session->GetInputCount();
  for (size_t i = 0; i < icount; ++i) {
    auto name = ort_->session->GetInputNameAllocated(i, alloc);
    ort_->input_name_strs.emplace_back(name.get());
  }
  size_t ocount = ort_->session->GetOutputCount();
  for (size_t i = 0; i < ocount; ++i) {
    auto name = ort_->session->GetOutputNameAllocated(i, alloc);
    ort_->output_name_strs.emplace_back(name.get());
  }
  ort_->input_count = icount;
}

void OnnxRunner::destroy_session_() {
  if (!ort_) return;
  ort_->session.reset();
  delete ort_;
  ort_ = nullptr;
}
#endif

OnnxRunner::OnnxRunner(const std::string& model_path)
    : model_path_(model_path) {}

std::pair<std::vector<double>, double> OnnxRunner::score_candidates(
    const ExtractedFeatures& feats) {
#ifndef BBML_WITH_ONNX
  return {std::vector<double>(feats.candidates.size(), 0.0), 0.0};
#else
  const size_t m = feats.candidates.size();
  if (model_path_.empty()) {
    return {std::vector<double>(m, 0.0), 0.0};
  }
  try {
    ensure_session_();
  } catch (const Ort::Exception&) {
    return {std::vector<double>(m, 0.0), 0.0};
  }
  Ort::AllocatorWithDefaultOptions alloc;
  // Build name arrays from stored strings
  std::vector<const char*> in_names, out_names;
  in_names.reserve(ort_->input_name_strs.size());
  out_names.reserve(ort_->output_name_strs.size());
  for (auto& s : ort_->input_name_strs) in_names.push_back(s.c_str());
  for (auto& s : ort_->output_name_strs) out_names.push_back(s.c_str());

  // number of candidates defines output size
  // const size_t m already defined above
  std::vector<Ort::Value> inputs;
  inputs.reserve(ort_->input_count);

  if (ort_->input_count <= 1) {
    // MLP or var-only GNN signature: single [m, d]
    const size_t d = 6;
    std::vector<float> x(m * d, 0.0f);
    for (size_t i = 0; i < m; ++i) {
      const auto& c = feats.candidates[i];
      x[i * d + 0] = static_cast<float>(c.obj);
      x[i * d + 1] = static_cast<float>(c.reduced_cost);
      x[i * d + 2] = static_cast<float>(c.fracval);
      x[i * d + 3] = static_cast<float>(c.domain_width);
      x[i * d + 4] = static_cast<float>(c.is_binary);
      x[i * d + 5] = static_cast<float>(c.is_integer);
    }
    std::array<int64_t, 2> shape{static_cast<int64_t>(m), 6};
    Ort::Value in0 = Ort::Value::CreateTensor<float>(alloc, shape.data(), shape.size());
    std::memcpy(in0.GetTensorMutableData<float>(), x.data(), x.size() * sizeof(float));
    inputs.emplace_back(std::move(in0));
  } else {
    // Graph signature: (var_feat [n_var,d_var], con_feat [n_con,d_con], edge_index [2,E])
    int n_var = feats.graph.n_var;
    int d_var = feats.graph.d_var;
    int n_con = feats.graph.n_con;
    int d_con = feats.graph.d_con;
    size_t E = feats.graph.edge_rows.size();
    if (n_var <= 0 || d_var <= 0) {
      // No graph features; return zeros
      return {std::vector<double>(m, 0.0), 0.0};
    }
    // var_feat
    std::array<int64_t, 2> vshape{static_cast<int64_t>(n_var), static_cast<int64_t>(d_var)};
    Ort::Value vfeat = Ort::Value::CreateTensor<float>(alloc, vshape.data(), vshape.size());
    {
      float* dst = vfeat.GetTensorMutableData<float>();
      size_t n = static_cast<size_t>(n_var) * static_cast<size_t>(d_var);
      for (size_t i = 0; i < n; ++i) dst[i] = static_cast<float>(feats.graph.var_feat[i]);
    }
    inputs.emplace_back(std::move(vfeat));
    // con_feat (ensure at least 1 row to avoid zero-dim issues)
    if (n_con <= 0) {
      n_con = 1; d_con = std::max(d_con, 1);
    }
    std::array<int64_t, 2> cshape{static_cast<int64_t>(n_con), static_cast<int64_t>(d_con)};
    Ort::Value cfeat = Ort::Value::CreateTensor<float>(alloc, cshape.data(), cshape.size());
    {
      float* dst = cfeat.GetTensorMutableData<float>();
      size_t n = static_cast<size_t>(n_con) * static_cast<size_t>(d_con);
      if (!feats.graph.con_feat.empty()) {
        size_t srcn = std::min(n, feats.graph.con_feat.size());
        for (size_t i = 0; i < srcn; ++i) dst[i] = static_cast<float>(feats.graph.con_feat[i]);
        for (size_t i = srcn; i < n; ++i) dst[i] = 0.0f;
      } else {
        std::memset(dst, 0, n * sizeof(float));
      }
    }
    inputs.emplace_back(std::move(cfeat));
    // edge_index [2, E]
    std::array<int64_t, 2> eshape{2, static_cast<int64_t>(E)};
    Ort::Value eidx = Ort::Value::CreateTensor<int64_t>(alloc, eshape.data(), eshape.size());
    if (E > 0) {
      int64_t* dst = eidx.GetTensorMutableData<int64_t>();
      for (size_t e = 0; e < E; ++e) {
        dst[e] = static_cast<int64_t>(feats.graph.edge_rows[e]);
      }
      for (size_t e = 0; e < E; ++e) {
        dst[E + e] = static_cast<int64_t>(feats.graph.edge_cols[e]);
      }
    }
    inputs.emplace_back(std::move(eidx));
  }

  std::vector<Ort::Value> outputs;
  try {
    outputs = ort_->session->Run(Ort::RunOptions{nullptr},
                                 in_names.data(), inputs.data(), inputs.size(),
                                 out_names.data(), out_names.size());
  } catch (const Ort::Exception&) {
    return {std::vector<double>(m, 0.0), 0.0};
  }
  // Assume single output scores [m]
  std::vector<double> scores(m, 0.0);
  if (!outputs.empty() && outputs[0].IsTensor()) {
    const float* s = outputs[0].GetTensorData<float>();
    for (size_t i = 0; i < m; ++i) scores[i] = static_cast<double>(s[i]);
  }
  return {std::move(scores), 0.5};
#endif
}

}  // namespace bbml
