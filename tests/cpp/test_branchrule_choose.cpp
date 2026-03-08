#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "bbml/branchrule_ml.hpp"

static bbml::ExtractedFeatures make_feats(
    std::initializer_list<double> redcosts) {
  bbml::ExtractedFeatures f{};
  int idx = 0;
  for (double rc : redcosts) {
    bbml::CandidateFeature c{};
    c.var_index = idx++;
    c.obj = 0.0;
    c.reduced_cost = rc;
    c.fracval = 0.0;
    c.domain_width = 1.0;
    c.is_binary = 1;
    c.is_integer = 1;
    f.candidates.push_back(c);
  }
  f.node.depth = 0;
  f.node.best_bound = 0.0;
  f.node.incumbent = 0.0;
  f.node.gap = 0.0;
  f.node.lp_time = 0.0;
  f.node.lp_iterations = 0;
  f.node.cut_rounds = 0;
  f.node.refactor_count = 0;
  f.node.cond_est = 0.0;
  return f;
}

TEST(BranchruleChoose, PicksLargestFallbackScoreWhenNoML) {
  auto runner = std::make_unique<bbml::OnnxRunner>(std::string(""));
  bbml::BranchruleML br(std::move(runner));
  auto feats = make_feats({-0.1, 0.5, -1.2, 0.9});
  std::vector<double> fallback_scores{0.1, 0.5, 1.2, 0.9};
  std::vector<double> blended;
  double alpha = 0.0;
  double used_confidence = -1.0;
  bool used_runtime_confidence = true;
  int idx = br.choose(feats,
                      fallback_scores,
                      &blended,
                      &alpha,
                      &used_confidence,
                      &used_runtime_confidence,
                      /*confidence=*/0.0,
                      /*depth=*/0);
  EXPECT_EQ(idx, 2);
  ASSERT_EQ(blended.size(), feats.candidates.size());
  // with confidence 0, alpha should be near lower bound (amin=0.1)
  EXPECT_GE(alpha, 0.0);
  EXPECT_DOUBLE_EQ(used_confidence, 0.0);
  EXPECT_FALSE(used_runtime_confidence);
}

TEST(BranchruleChoose, RaisesAlphaWhenConfidenceExceedsTheta) {
  auto runner = std::make_unique<bbml::OnnxRunner>(std::string(""));
  bbml::BranchruleML br(std::move(runner));
  br.set_alpha_params(0.1, 0.8, 0.0);
  br.set_alpha_theta(0.5);

  auto feats = make_feats({0.1, 0.2});
  std::vector<double> fallback_scores{0.2, 0.1};
  std::vector<double> blended;
  double alpha_low = 0.0;
  double alpha_high = 0.0;
  double used_confidence = 0.0;
  bool used_runtime_confidence = false;

  (void)br.choose(feats,
                  fallback_scores,
                  &blended,
                  &alpha_low,
                  &used_confidence,
                  &used_runtime_confidence,
                  /*confidence=*/0.2,
                  /*depth=*/0);
  (void)br.choose(feats,
                  fallback_scores,
                  &blended,
                  &alpha_high,
                  &used_confidence,
                  &used_runtime_confidence,
                  /*confidence=*/0.8,
                  /*depth=*/0);

  EXPECT_LT(alpha_low, alpha_high);
}
