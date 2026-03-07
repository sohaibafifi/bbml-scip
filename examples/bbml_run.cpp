#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <utility>
#include <iostream>
#include <cstdint>
#include <cinttypes>
#include <algorithm>

#include "scip/scip.h"
#include "scip/scipdefplugins.h"
#include "scip/pub_paramset.h"
#include "bbml/branchrule_ml.hpp"
#include "bbml/nodesel_ml.hpp"

namespace {

bool parse_bool_value(const std::string& val, SCIP_Bool* out) {
  if (val == "TRUE" || val == "true" || val == "1") {
    *out = TRUE;
    return true;
  }
  if (val == "FALSE" || val == "false" || val == "0") {
    *out = FALSE;
    return true;
  }
  return false;
}

void strip_quotes(std::string* val) {
  if (val->size() >= 2 &&
      ((val->front() == '"' && val->back() == '"') ||
       (val->front() == '\'' && val->back() == '\''))) {
    *val = val->substr(1, val->size() - 2);
  }
}

bool set_param(SCIP* scip, const std::string& kv) {
  auto pos = kv.find('=');
  if (pos == std::string::npos) {
    std::cerr << "Invalid --param (expected name=value): " << kv << "\n";
    return false;
  }
  std::string name = kv.substr(0, pos);
  std::string val = kv.substr(pos + 1);
  // trim spaces
  auto trim = [](std::string& x) {
    size_t a = x.find_first_not_of(" \t\n\r");
    size_t b = x.find_last_not_of(" \t\n\r");
    if (a == std::string::npos) {
      x.clear();
      return;
    }
    x = x.substr(a, b - a + 1);
  };
  trim(name);
  trim(val);
  SCIP_PARAM* param = SCIPgetParam(scip, name.c_str());
  if (param == nullptr) {
    std::cerr << "Unknown SCIP parameter: " << name << "\n";
    return false;
  }

  switch (SCIPparamGetType(param)) {
    case SCIP_PARAMTYPE_BOOL: {
      SCIP_Bool b = FALSE;
      if (!parse_bool_value(val, &b)) {
        std::cerr << "Invalid bool value for " << name << ": " << val << "\n";
        return false;
      }
      return SCIPsetBoolParam(scip, name.c_str(), b) == SCIP_OKAY;
    }
    case SCIP_PARAMTYPE_INT: {
      char* end = nullptr;
      long parsed = strtol(val.c_str(), &end, 10);
      if (!(end && *end == '\0')) {
        std::cerr << "Invalid int value for " << name << ": " << val << "\n";
        return false;
      }
      return SCIPsetIntParam(scip, name.c_str(), static_cast<int>(parsed)) == SCIP_OKAY;
    }
    case SCIP_PARAMTYPE_LONGINT: {
      char* end = nullptr;
      int64_t parsed = static_cast<int64_t>(strtoll(val.c_str(), &end, 10));
      if (!(end && *end == '\0')) {
        std::cerr << "Invalid longint value for " << name << ": " << val << "\n";
        return false;
      }
      return SCIPsetLongintParam(scip, name.c_str(), static_cast<SCIP_Longint>(parsed)) == SCIP_OKAY;
    }
    case SCIP_PARAMTYPE_REAL: {
      char* end = nullptr;
      double parsed = strtod(val.c_str(), &end);
      if (!(end && *end == '\0')) {
        std::cerr << "Invalid real value for " << name << ": " << val << "\n";
        return false;
      }
      return SCIPsetRealParam(scip, name.c_str(), parsed) == SCIP_OKAY;
    }
    case SCIP_PARAMTYPE_CHAR: {
      strip_quotes(&val);
      if (val.empty()) {
        std::cerr << "Invalid char value for " << name << ": " << val << "\n";
        return false;
      }
      return SCIPsetCharParam(scip, name.c_str(), val[0]) == SCIP_OKAY;
    }
    case SCIP_PARAMTYPE_STRING: {
      strip_quotes(&val);
      return SCIPsetStringParam(scip, name.c_str(), const_cast<char*>(val.c_str())) == SCIP_OKAY;
    }
  }
  std::cerr << "Unsupported parameter type for " << name << "\n";
  return false;
}

void usage() {
  std::cerr << "Usage: bbml_run --problem <file> [--set file.set]* [--param name=value]*\n";
}

}  // namespace

int main(int argc, char** argv) {
  std::string plugin, problem;
  std::vector<std::string> setfiles;
  std::vector<std::string> kvparams;
  for (int i = 1; i < argc; ++i) {
    std::string a(argv[i]);
    if (a == "--plugin" && i + 1 < argc) {
      plugin = argv[++i];
    } else if (a == "--problem" && i + 1 < argc) {
      problem = argv[++i];
    } else if (a == "--set" && i + 1 < argc) {
      setfiles.emplace_back(argv[++i]);
    } else if (a == "--param" && i + 1 < argc) {
      kvparams.emplace_back(argv[++i]);
    } else {
      std::cerr << "Unknown or incomplete arg: " << a << "\n";
      usage();
      return 2;
    }
  }
  if (problem.empty()) {
    usage();
    return 2;
  }
  if (!plugin.empty()) {
    std::cerr << "Warning: --plugin is ignored; bbml_run already includes the BBML plugins.\n";
  }

  SCIP* scip = nullptr;
  if (SCIPcreate(&scip) != SCIP_OKAY) {
    std::cerr << "SCIPcreate failed" << std::endl;
    return 1;
  }
  SCIPincludeDefaultPlugins(scip);
  // Directly include bbml plugins (statically linked)
  if (bbml::includeBranchruleML(scip) != SCIP_OKAY) {
    std::cerr << "includeBranchruleML failed" << std::endl;
    return 1;
  }
  if (bbml::includeNodeselML(scip) != SCIP_OKAY) {
    std::cerr << "includeNodeselML failed" << std::endl;
    return 1;
  }

  // Apply inline params
  for (const auto& kv : kvparams) {
    if (!set_param(scip, kv)) {
      SCIPfree(&scip);
      return 2;
    }
  }
  // Apply .set files using SCIP native parser
  for (const auto& sf : setfiles) {
    (void)SCIPreadParams(scip, sf.c_str());
  }

  if (SCIPcreateProbBasic(scip, "bbml_problem") != SCIP_OKAY) {
    std::cerr << "SCIPcreateProbBasic failed" << std::endl;
    return 1;
  }
  if (SCIPreadProb(scip, problem.c_str(), nullptr) != SCIP_OKAY) {
    std::cerr << "SCIPreadProb failed" << std::endl;
    return 1;
  }
  if (SCIPsolve(scip) != SCIP_OKAY) {
    std::cerr << "SCIPsolve failed" << std::endl;
    return 1;
  }
  // Print solving statistics
  SCIPprintStatistics(scip, nullptr);

  // Machine-readable summary for downstream parsing
  SCIP_Longint nnodes = SCIPgetNNodes(scip);
  SCIP_Real time = SCIPgetSolvingTime(scip);
  SCIP_Real gap = SCIPgetGap(scip);
  // Some SCIP versions expose only primal-dual integral; use it as proxy
  SCIP_Real pInt = SCIPgetPrimalDualIntegral(scip);
  SCIP_STATUS status = SCIPgetStatus(scip);
  int solved = (status == SCIP_STATUS_OPTIMAL) ? 1 : 0;
  const int64_t nnodes_i64 = static_cast<int64_t>(nnodes);
  std::printf(
      "BBML_SUMMARY nodes=%" PRId64 " time=%.9f gap=%.9f primal_integral=%.9f solved=%d status=%d\n",
      nnodes_i64, static_cast<double>(time),
      static_cast<double>(gap), static_cast<double>(pInt), solved, static_cast<int>(status));

  SCIPfree(&scip);
  return 0;
}
