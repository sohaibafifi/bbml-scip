#include "bbml/onnx_runner.hpp"
#include <limits>
#include <cmath>
#include <sstream>
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

namespace {

std::string trimCopy(std::string s) {
  const auto begin = s.find_first_not_of(" \t\n\r");
  if (begin == std::string::npos) {
    return std::string();
  }
  const auto end = s.find_last_not_of(" \t\n\r");
  return s.substr(begin, end - begin + 1);
}

std::vector<std::string> parseModelPathSpec(const std::string& spec) {
  std::vector<std::string> paths;
  std::stringstream ss(spec);
  std::string item;
  while (std::getline(ss, item, ',')) {
    item = trimCopy(std::move(item));
    if (!item.empty()) {
      paths.push_back(item);
    }
  }
  if (paths.empty()) {
    const std::string trimmed = trimCopy(spec);
    if (!trimmed.empty()) {
      paths.push_back(trimmed);
    }
  }
  return paths;
}

void fillSingleInputFeatures(const CandidateFeature& c, float* dst, size_t d) {
  std::fill(dst, dst + d, 0.0f);
  if (d > 0) dst[0] = static_cast<float>(c.obj);
  if (d > 1) dst[1] = static_cast<float>(c.reduced_cost);
  if (d > 2) dst[2] = static_cast<float>(c.fracval);
  if (d > 3) dst[3] = static_cast<float>(c.domain_width);
  if (d > 4) dst[4] = static_cast<float>(c.is_binary);
  if (d > 5) dst[5] = static_cast<float>(c.is_integer);
  if (d > 6) dst[6] = static_cast<float>(c.pseudocost_up);
  if (d > 7) dst[7] = static_cast<float>(c.pseudocost_down);
  if (d > 8) dst[8] = static_cast<float>(c.pc_obs_up);
  if (d > 9) dst[9] = static_cast<float>(c.pc_obs_down);
  if (d > 10) dst[10] = static_cast<float>(c.at_lb);
  if (d > 11) dst[11] = static_cast<float>(c.at_ub);
  if (d > 12) dst[12] = static_cast<float>(c.col_nnz);
}

}  // namespace

#ifdef BBML_WITH_ONNX
struct OnnxRunner::OrtHolder {
  struct SessionHolder {
    Ort::SessionOptions opts;
    std::unique_ptr<Ort::Session> session;
    std::vector<std::string> input_name_strs;
    std::vector<std::string> output_name_strs;
    size_t input_dim = 0;
    size_t input_count = 0;
  };

  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "bbml"};
  std::vector<SessionHolder> sessions;
  std::string loaded_model_spec;
};

void OnnxRunner::ensure_session_() {
  if (!ort_) ort_ = new OrtHolder();
  if (!ort_->sessions.empty() && ort_->loaded_model_spec == model_path_) {
    return;
  }

  ort_->sessions.clear();
  ort_->loaded_model_spec = model_path_;

  const auto model_paths = parseModelPathSpec(model_path_);
  Ort::AllocatorWithDefaultOptions alloc;
  ort_->sessions.reserve(model_paths.size());
  for (const auto& path : model_paths) {
    OrtHolder::SessionHolder holder;
    holder.opts.SetIntraOpNumThreads(1);
    holder.opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);
    holder.session.reset(new Ort::Session(ort_->env, path.c_str(), holder.opts));

    const size_t icount = holder.session->GetInputCount();
    for (size_t i = 0; i < icount; ++i) {
      auto name = holder.session->GetInputNameAllocated(i, alloc);
      holder.input_name_strs.emplace_back(name.get());
    }
    const size_t ocount = holder.session->GetOutputCount();
    for (size_t i = 0; i < ocount; ++i) {
      auto name = holder.session->GetOutputNameAllocated(i, alloc);
      holder.output_name_strs.emplace_back(name.get());
    }
    holder.input_count = icount;
    holder.input_dim = 0;
    if (icount == 1) {
      auto type_info = holder.session->GetInputTypeInfo(0);
      auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
      auto shape = tensor_info.GetShape();
      if (shape.size() >= 2 && shape[1] > 0) {
        holder.input_dim = static_cast<size_t>(shape[1]);
      }
    }
    ort_->sessions.emplace_back(std::move(holder));
  }
}

void OnnxRunner::destroy_session_() {
  if (!ort_) return;
  ort_->sessions.clear();
  delete ort_;
  ort_ = nullptr;
}
#endif

OnnxRunner::OnnxRunner(const std::string& model_path)
    : model_path_(model_path) {}

OnnxRunner::~OnnxRunner() {
#ifdef BBML_WITH_ONNX
  destroy_session_();
#endif
}

bool OnnxRunner::requires_graph_inputs() {
#ifndef BBML_WITH_ONNX
  return false;
#else
  if (model_path_.empty()) {
    return false;
  }
  try {
    ensure_session_();
  } catch (const Ort::Exception&) {
    return false;
  }
  if (ort_ == nullptr || ort_->sessions.empty()) {
    return false;
  }
  return ort_->sessions.front().input_count > 1;
#endif
}

std::pair<std::vector<double>, double> OnnxRunner::score_candidates(
    const ExtractedFeatures& feats) {
#ifndef BBML_WITH_ONNX
  return {std::vector<double>(feats.candidates.size(), 0.0),
          std::numeric_limits<double>::quiet_NaN()};
#else
  const size_t m = feats.candidates.size();
  if (model_path_.empty()) {
    return {std::vector<double>(m, 0.0), std::numeric_limits<double>::quiet_NaN()};
  }
  try {
    ensure_session_();
  } catch (const Ort::Exception&) {
    return {std::vector<double>(m, 0.0), std::numeric_limits<double>::quiet_NaN()};
  }

  if (ort_->sessions.empty()) {
    return {std::vector<double>(m, 0.0), std::numeric_limits<double>::quiet_NaN()};
  }

  std::vector<std::vector<double>> member_scores;
  member_scores.reserve(ort_->sessions.size());
  try {
    Ort::AllocatorWithDefaultOptions alloc;
    for (auto& holder : ort_->sessions) {
      std::vector<const char*> in_names, out_names;
      in_names.reserve(holder.input_name_strs.size());
      out_names.reserve(holder.output_name_strs.size());
      for (auto& s : holder.input_name_strs) in_names.push_back(s.c_str());
      for (auto& s : holder.output_name_strs) out_names.push_back(s.c_str());

      std::vector<Ort::Value> inputs;
      inputs.reserve(holder.input_count);

      if (holder.input_count <= 1) {
        const size_t d = holder.input_dim > 0 ? holder.input_dim : 10;
        std::vector<float> x(m * d, 0.0f);
        for (size_t i = 0; i < m; ++i) {
          fillSingleInputFeatures(feats.candidates[i], x.data() + (i * d), d);
        }
        std::array<int64_t, 2> shape{static_cast<int64_t>(m), static_cast<int64_t>(d)};
        Ort::Value in0 = Ort::Value::CreateTensor<float>(alloc, shape.data(), shape.size());
        std::memcpy(in0.GetTensorMutableData<float>(), x.data(), x.size() * sizeof(float));
        inputs.emplace_back(std::move(in0));
      } else {
        int n_var = feats.graph.n_var;
        int d_var = feats.graph.d_var;
        int n_con = feats.graph.n_con;
        int d_con = feats.graph.d_con;
        const size_t E = feats.graph.edge_rows.size();
        if (n_var <= 0 || d_var <= 0) {
          return {std::vector<double>(m, 0.0), std::numeric_limits<double>::quiet_NaN()};
        }
        std::array<int64_t, 2> vshape{static_cast<int64_t>(n_var), static_cast<int64_t>(d_var)};
        Ort::Value vfeat = Ort::Value::CreateTensor<float>(alloc, vshape.data(), vshape.size());
        {
          float* dst = vfeat.GetTensorMutableData<float>();
          const size_t n = static_cast<size_t>(n_var) * static_cast<size_t>(d_var);
          for (size_t i = 0; i < n; ++i) dst[i] = static_cast<float>(feats.graph.var_feat[i]);
        }
        inputs.emplace_back(std::move(vfeat));

        if (n_con <= 0) {
          n_con = 1;
          d_con = std::max(d_con, 1);
        }
        std::array<int64_t, 2> cshape{static_cast<int64_t>(n_con), static_cast<int64_t>(d_con)};
        Ort::Value cfeat = Ort::Value::CreateTensor<float>(alloc, cshape.data(), cshape.size());
        {
          float* dst = cfeat.GetTensorMutableData<float>();
          const size_t n = static_cast<size_t>(n_con) * static_cast<size_t>(d_con);
          if (!feats.graph.con_feat.empty()) {
            const size_t srcn = std::min(n, feats.graph.con_feat.size());
            for (size_t i = 0; i < srcn; ++i) dst[i] = static_cast<float>(feats.graph.con_feat[i]);
            for (size_t i = srcn; i < n; ++i) dst[i] = 0.0f;
          } else {
            std::memset(dst, 0, n * sizeof(float));
          }
        }
        inputs.emplace_back(std::move(cfeat));

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

      auto outputs = holder.session->Run(Ort::RunOptions{nullptr},
                                         in_names.data(),
                                         inputs.data(),
                                         inputs.size(),
                                         out_names.data(),
                                         out_names.size());
      std::vector<double> scores(m, 0.0);
      if (!outputs.empty() && outputs[0].IsTensor()) {
        const float* s = outputs[0].GetTensorData<float>();
        for (size_t i = 0; i < m; ++i) scores[i] = static_cast<double>(s[i]);
      }
      member_scores.emplace_back(std::move(scores));
    }
  } catch (const Ort::Exception&) {
    return {std::vector<double>(m, 0.0), std::numeric_limits<double>::quiet_NaN()};
  }

  if (member_scores.empty()) {
    return {std::vector<double>(m, 0.0), std::numeric_limits<double>::quiet_NaN()};
  }

  std::vector<double> mean_scores(m, 0.0);
  for (const auto& scores : member_scores) {
    for (size_t i = 0; i < m; ++i) {
      mean_scores[i] += scores[i];
    }
  }
  const double inv_member_count = 1.0 / static_cast<double>(member_scores.size());
  for (double& score : mean_scores) {
    score *= inv_member_count;
  }

  if (member_scores.size() == 1) {
    return {std::move(mean_scores), std::numeric_limits<double>::quiet_NaN()};
  }

  double max_std = 0.0;
  double mean_abs = 0.0;
  for (size_t i = 0; i < m; ++i) {
    double var = 0.0;
    for (const auto& scores : member_scores) {
      const double diff = scores[i] - mean_scores[i];
      var += diff * diff;
    }
    var *= inv_member_count;
    max_std = std::max(max_std, std::sqrt(std::max(0.0, var)));
    mean_abs += std::fabs(mean_scores[i]);
  }
  mean_abs /= std::max<size_t>(1, m);
  const double scale = std::max(1e-3, mean_abs);
  const double confidence = std::clamp(scale / (scale + max_std), 0.0, 1.0);
  return {std::move(mean_scores), confidence};
#endif
}

}  // namespace bbml
