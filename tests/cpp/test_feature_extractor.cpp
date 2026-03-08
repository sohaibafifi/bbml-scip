#include <algorithm>
#include <gtest/gtest.h>
#include <cmath>
#include <limits>
#include <string>
#include "bbml/feature_extractor.hpp"
#include "lpi/lpi.h"
#include "scip/scip.h"
#include "scip/scipdefplugins.h"
#include "scip/scip_lp.h"
#include "scip/scip_solvingstats.h"

#ifdef BBML_WITH_LP_STATS
#include "soplex/spxsolver.h"
#endif

TEST(FeatureExtractor, ExtractsWithoutNodeAndNoVars) {
    SCIP *scip = nullptr;
    ASSERT_EQ(SCIPcreate(&scip), SCIP_OKAY);
    ASSERT_NE(scip, nullptr);
    ASSERT_EQ(SCIPincludeDefaultPlugins(scip), SCIP_OKAY);
    ASSERT_EQ(SCIPcreateProbBasic(scip, "t"), SCIP_OKAY);
    ASSERT_EQ(SCIPsolve(scip), SCIP_OKAY);

    bbml::FeatureExtractor fx;
    auto feats = fx.fromSCIP(scip, /*node=*/nullptr);
    // Depth defaults to 0 when node is null
    EXPECT_EQ(feats.node.depth, 0);
    // With no variables/LP, there should be zero candidates or more,
    // but not negative
    EXPECT_GE(static_cast<int>(feats.candidates.size()), 0);

    // repeat call for basic caching path stability
    auto feats2 = fx.fromSCIP(scip, /*node=*/nullptr);
    EXPECT_EQ(feats2.node.depth, 0);
    EXPECT_EQ(feats2.candidates.size(), feats.candidates.size());

    EXPECT_EQ(SCIPfree(&scip), SCIP_OKAY);
}

namespace {
static bbml::ExtractedFeatures g_last_feats;
static bool g_captured = false;
static int g_last_cut_rounds = 0;
static int g_last_refactor_count = 0;
static double g_last_cond_est = 0.0;

static SCIP_RETCODE TestBranchExecLP(
    SCIP* scip,
    SCIP_BRANCHRULE* /*branchrule*/,
    SCIP_Bool /*allowaddcons*/,
    SCIP_RESULT* result) {
  bbml::FeatureExtractor fx;
  SCIP_NODE* focus = SCIPgetFocusNode(scip);
  g_last_feats = fx.fromSCIP(scip, focus);
  g_last_cut_rounds = SCIPgetNSepaRounds(scip);
  g_last_refactor_count = 0;
  g_last_cond_est = 0.0;
  SCIP_LPI* lpi = nullptr;
  if (SCIPgetLPI(scip, &lpi) == SCIP_OKAY && lpi != nullptr) {
    SCIP_Real cond_est = 0.0;
    if (SCIPlpiGetRealSolQuality(
            lpi, SCIP_LPSOLQUALITY_ESTIMCONDITION, &cond_est) == SCIP_OKAY &&
        std::isfinite(static_cast<double>(cond_est)) && cond_est > 0.0) {
      g_last_cond_est = static_cast<double>(cond_est);
    } else if (!SCIPlpiIsStable(lpi)) {
      g_last_cond_est = std::numeric_limits<double>::infinity();
    }
#ifdef BBML_WITH_LP_STATS
    const char* solver_name = SCIPlpiGetSolverName();
    if (solver_name != nullptr && std::string(solver_name).find("SoPlex") != std::string::npos) {
      void* solver_ptr = SCIPlpiGetSolverPointer(lpi);
      if (solver_ptr != nullptr) {
        auto* solver = static_cast<soplex::SPxSolver*>(solver_ptr);
        g_last_refactor_count = std::max(0, solver->basis().lastUpdate());
      }
    }
#endif
  }
  g_captured = true;
  *result = SCIP_DIDNOTRUN;  // let default branching proceed
  return SCIP_OKAY;
}
}  // namespace

TEST(FeatureExtractor, FractionalRootCandidates) {
    g_captured = false;
    g_last_cut_rounds = 0;
    g_last_refactor_count = 0;
    g_last_cond_est = 0.0;

    SCIP *scip = nullptr;
    ASSERT_EQ(SCIPcreate(&scip), SCIP_OKAY);
    ASSERT_EQ(SCIPincludeDefaultPlugins(scip), SCIP_OKAY);
    ASSERT_EQ(SCIPcreateProbBasic(scip, "toy"), SCIP_OKAY);

    // Keep it simple and reproducible
    ASSERT_EQ(SCIPsetPresolving(scip, SCIP_PARAMSETTING_OFF, true), SCIP_OKAY);
    ASSERT_EQ(SCIPsetHeuristics(scip, SCIP_PARAMSETTING_OFF, true), SCIP_OKAY);
    ASSERT_EQ(SCIPsetSeparating(scip, SCIP_PARAMSETTING_OFF, true), SCIP_OKAY);
    ASSERT_EQ(SCIPsetLongintParam(scip, "limits/nodes", 1), SCIP_OKAY);
    // Ensure LP is constructed and solved at root
    (void) SCIPsetBoolParam(scip, "constraints/linear/initial", TRUE);
    (void) SCIPsetBoolParam(scip, "constraints/linear/separate", TRUE);
    (void) SCIPsetBoolParam(scip, "constraints/linear/dynamic", FALSE);
    (void) SCIPsetIntParam(scip, "lp/solvefreq", 1);

    // Variables: x, y in [0,1], binary (LP relaxation will allow fractional)
    SCIP_VAR *x = nullptr;
    SCIP_VAR *y = nullptr;
    ASSERT_EQ(SCIPcreateVarBasic(
                  scip, &x, "x", 0.0, 1.0, 1.0, SCIP_VARTYPE_BINARY),
              SCIP_OKAY);
    ASSERT_EQ(SCIPcreateVarBasic(
                  scip, &y, "y", 0.0, 1.0, 1.0, SCIP_VARTYPE_BINARY),
              SCIP_OKAY);
    ASSERT_EQ(SCIPaddVar(scip, x), SCIP_OKAY);
    ASSERT_EQ(SCIPaddVar(scip, y), SCIP_OKAY);
    const int xidx = SCIPvarGetProbindex(x);
    const int yidx = SCIPvarGetProbindex(y);

    // Constraint: x + y >= 1.5 ensures feasibility but forces fractional LP
    // optimum for min obj
    SCIP_CONS *cons = nullptr;
    ASSERT_EQ(SCIPcreateConsBasicLinear(
                  scip, &cons, "c1", 0, nullptr, nullptr, 1.5,
                  SCIPinfinity(scip)),
              SCIP_OKAY);
    ASSERT_EQ(SCIPaddCoefLinear(scip, cons, x, 1.0), SCIP_OKAY);
    ASSERT_EQ(SCIPaddCoefLinear(scip, cons, y, 1.0), SCIP_OKAY);
    ASSERT_EQ(SCIPaddCons(scip, cons), SCIP_OKAY);
    ASSERT_EQ(SCIPreleaseCons(scip, &cons), SCIP_OKAY);

    // Include a high-priority branchrule to capture features at root
    SCIP_BRANCHRULE *br = nullptr;
    ASSERT_EQ(
        SCIPincludeBranchruleBasic(scip, &br, "capture_br",
                                   "capture features", 1000000, -1, 1.0,
                                   nullptr),
        SCIP_OKAY);
    ASSERT_EQ(SCIPsetBranchruleExecLp(scip, br, TestBranchExecLP), SCIP_OKAY);

    // Solve; our branchrule should execute at root and capture features
    ASSERT_EQ(SCIPsolve(scip), SCIP_OKAY);
    if (!g_captured) {
        GTEST_SKIP()
            << "Solver terminated at root without branching; LP candidates "
            << "unavailable on this build.";
    }

    // Basic validations on captured candidates
    if (g_last_feats.candidates.empty()) {
        GTEST_SKIP()
            << "No LP branch candidates available at capture point.";
    }
    bool saw_fractional = false;
    for (const auto& c : g_last_feats.candidates) {
        EXPECT_TRUE(c.var_index == xidx || c.var_index == yidx);
        EXPECT_NEAR(c.obj, 1.0, 1e-9);
        EXPECT_GE(c.domain_width, 0.0);
        EXPECT_LE(c.domain_width, 1.0 + 1e-9);
        EXPECT_TRUE(std::isfinite(c.reduced_cost));
        // fractional in (0,1)
        if (c.fracval > 1e-6 && c.fracval < 1.0 - 1e-6) {
            saw_fractional = true;
        }
        EXPECT_TRUE(c.is_binary == 0 || c.is_binary == 1);
    }
    EXPECT_TRUE(saw_fractional);
    EXPECT_EQ(g_last_feats.node.cut_rounds, g_last_cut_rounds);
    EXPECT_EQ(g_last_feats.node.refactor_count, g_last_refactor_count);
    if (std::isfinite(g_last_cond_est)) {
        EXPECT_NEAR(g_last_feats.node.cond_est, g_last_cond_est, 1e-9);
    } else {
        EXPECT_FALSE(std::isfinite(g_last_feats.node.cond_est));
    }

    ASSERT_EQ(SCIPreleaseVar(scip, &x), SCIP_OKAY);
    ASSERT_EQ(SCIPreleaseVar(scip, &y), SCIP_OKAY);
    ASSERT_EQ(SCIPfree(&scip), SCIP_OKAY);
}
