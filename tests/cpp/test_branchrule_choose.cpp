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

TEST(BranchruleChoose, PicksLargestAbsReducedCostWhenNoML) {
  auto runner = std::make_unique<bbml::OnnxRunner>(std::string(""));
  bbml::BranchruleML br(std::move(runner));
  auto feats = make_feats({-0.1, 0.5, -1.2, 0.9});
  std::vector<double> blended;
  double alpha = 0.0;
  int idx = br.choose(feats, &blended, &alpha, /*confidence=*/0.0,
                      /*depth=*/0);
  EXPECT_EQ(idx, 2);  // |-1.2| is largest
  ASSERT_EQ(blended.size(), feats.candidates.size());
  // with confidence 0, alpha should be near lower bound (amin=0.1)
  EXPECT_GE(alpha, 0.0);
}
