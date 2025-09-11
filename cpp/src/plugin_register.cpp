#include "bbml/branchrule_ml.hpp"
#include "bbml/nodesel_ml.hpp"
#include "bbml/telemetry_logger.hpp"

extern "C" {
SCIP_RETCODE SCIPincludeBbmlPlugins(SCIP* scip) {
  SCIP_CALL(bbml::includeBranchruleML(scip));
  SCIP_CALL(bbml::includeNodeselML(scip));
  bbml::getTelemetryLogger().log_root(scip);
  return SCIP_OKAY;
}

// Provide alternative symbol names so `scip -l <lib>` can resolve the include
// function based on different basename mappings.
SCIP_RETCODE SCIPincludeBbml_scipPlugins(SCIP* scip) {
  return SCIPincludeBbmlPlugins(scip);
}
SCIP_RETCODE SCIPincludeBbmlScipPlugins(SCIP* scip) {
  return SCIPincludeBbmlPlugins(scip);
}
}
