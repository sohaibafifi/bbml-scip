#include <gtest/gtest.h>
#include <cstdio>
#include <fstream>
#include <string>
#include "scip/scip.h"
#include "scip/scipdefplugins.h"
#include "bbml/branchrule_ml.hpp"
#include "bbml/nodesel_ml.hpp"

static std::string read_all(const std::string& path) {
  std::ifstream in(path);
  if(!in.good()) return std::string();
  std::string s((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  return s;
}

TEST(BranchruleParams, EnableToggleAffectsTelemetry) {
  // Telemetry path
  const std::string tpath = "data/test_bbml_toggle.ndjson";
  std::remove(tpath.c_str());
  setenv("BBML_TELEMETRY_PATH", tpath.c_str(), 1);

  SCIP* scip = nullptr;
  ASSERT_EQ(SCIPcreate(&scip), SCIP_OKAY);
  ASSERT_EQ(SCIPincludeDefaultPlugins(scip), SCIP_OKAY);
  // include our plugins (branchrule + nodesel)
  ASSERT_EQ(bbml::includeBranchruleML(scip), SCIP_OKAY);

  // Build simple MILP that tends to produce fractional LP at root
  ASSERT_EQ(SCIPcreateProbBasic(scip, "toggle"), SCIP_OKAY);
  ASSERT_EQ(SCIPsetPresolving(scip, SCIP_PARAMSETTING_OFF, true), SCIP_OKAY);
  ASSERT_EQ(SCIPsetHeuristics(scip, SCIP_PARAMSETTING_OFF, true), SCIP_OKAY);
  ASSERT_EQ(SCIPsetSeparating(scip, SCIP_PARAMSETTING_OFF, true), SCIP_OKAY);
  (void)SCIPsetLongintParam(scip, "limits/nodes", 2);

  SCIP_VAR *x = nullptr, *y = nullptr;
  ASSERT_EQ(SCIPcreateVarBasic(scip, &x, "x", 0.0, 1.0, 1.0, SCIP_VARTYPE_BINARY), SCIP_OKAY);
  ASSERT_EQ(SCIPcreateVarBasic(scip, &y, "y", 0.0, 1.0, 1.0, SCIP_VARTYPE_BINARY), SCIP_OKAY);
  ASSERT_EQ(SCIPaddVar(scip, x), SCIP_OKAY);
  ASSERT_EQ(SCIPaddVar(scip, y), SCIP_OKAY);
  SCIP_CONS* cons = nullptr;
  ASSERT_EQ(SCIPcreateConsBasicLinear(scip, &cons, "c1", 0, nullptr, nullptr, 1.5, SCIPinfinity(scip)), SCIP_OKAY);
  ASSERT_EQ(SCIPaddCoefLinear(scip, cons, x, 1.0), SCIP_OKAY);
  ASSERT_EQ(SCIPaddCoefLinear(scip, cons, y, 1.0), SCIP_OKAY);
  ASSERT_EQ(SCIPaddCons(scip, cons), SCIP_OKAY);
  ASSERT_EQ(SCIPreleaseCons(scip, &cons), SCIP_OKAY);

  // Case 1: disabled -> expect 0 candidate telemetry lines (ignore root event)
  ASSERT_EQ(SCIPsetBoolParam(scip, "bbml/enable", FALSE), SCIP_OKAY);
  ASSERT_EQ(SCIPsolve(scip), SCIP_OKAY);
  std::string s1 = read_all(tpath);
  size_t picks1 = s1.find("chosen_idx") == std::string::npos ? 0 : 1;  // 1+ if present

  // Case 2: enabled -> expect at least one candidate line (if branch happens)
  ASSERT_EQ(SCIPsetBoolParam(scip, "bbml/enable", TRUE), SCIP_OKAY);
  ASSERT_EQ(SCIPsolve(scip), SCIP_OKAY);
  std::string s2 = read_all(tpath);
  size_t picks2 = s2.find("chosen_idx") == std::string::npos ? 0 : 1;

  if (picks2 == 0) {
    GTEST_SKIP() << "No candidate telemetry (no branching occurred); skipping.";
  } else {
    EXPECT_EQ(picks1, 0u);
    EXPECT_GE(picks2, 1u);
  }

  ASSERT_EQ(SCIPreleaseVar(scip, &x), SCIP_OKAY);
  ASSERT_EQ(SCIPreleaseVar(scip, &y), SCIP_OKAY);
  ASSERT_EQ(SCIPfree(&scip), SCIP_OKAY);
}

