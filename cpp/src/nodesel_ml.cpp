#include "bbml/nodesel_ml.hpp"

namespace {
// Comparator with explicit signature (no macro)
static int NodeSelComp(
    SCIP* scip,
    SCIP_NODESEL* nodesel,
    SCIP_NODE* node1,
    SCIP_NODE* node2) {
  const SCIP_Real lb1 = SCIPnodeGetLowerbound(node1);
  const SCIP_Real lb2 = SCIPnodeGetLowerbound(node2);
  if (lb1 < lb2) return -1;
  if (lb1 > lb2) return 1;
  return 0;
}

// Selector with explicit signature (no macro)
static SCIP_RETCODE NodeSelSelect(
    SCIP* scip,
    SCIP_NODESEL* nodesel,
    SCIP_NODE** selnode) {
  SCIP_NODE* best = SCIPgetBestboundNode(scip);
  if (best == nullptr) {
    best = SCIPgetBestNode(scip);
  }
  *selnode = best;
  return SCIP_OKAY;
}
}  // namespace

SCIP_RETCODE bbml::includeNodeselML(SCIP *scip) {
  SCIP_NODESEL *ns = nullptr;
  SCIP_CALL(SCIPincludeNodeselBasic(
    scip, &ns, "bbml_nodesel", "ML tiebreak node selector (best-bound)",
    0, 0, NodeSelSelect, NodeSelComp, nullptr));
  return SCIP_OKAY;
}
