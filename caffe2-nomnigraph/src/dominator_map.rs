crate::ix!();

/**
  | -----------
  | @brief
  | 
  | Map all nodes in the graph to their immediate
  | dominators.
  |
  */
#[inline] pub fn immediate_dominator_map<G: GraphType>(
    g:      *mut G, 
    source: Option<<G as GraphType>::NodeRef>) -> HashMap<<G as GraphType>::NodeRef,<G as GraphType>::NodeRef> 
{
    todo!();
    /*
        std::unordered_map<typename G::NodeRef, typename G::NodeRef> idomMap;
      auto idomTree = dominatorTree(g, source);
      for (auto node : idomTree.getMutableNodes()) {
        // Sanity check, really should never happen.
        assert(
            node->getInEdges().size() <= 1 &&
            "Invalid dominator tree generated from graph, cannot determing idom map.");
        // In degenerate cases, or for the root node, we self dominate.
        if (node->getInEdges().size() == 0) {
          idomMap[node->data()] = node->data();
        } else {
          auto idom = node->getInEdges()[0]->tail();
          idomMap[node->data()] = idom->data();
        }
      }
      return idomMap;
    */
}

/**
  | \brief Map all nodes to their dominance
  | frontiers: a set of nodes that does not
  | strictly dominate the given node but does
  | dominate an immediate predecessor.  This is
  | useful as it is the exact location for the
  | insertion of phi nodes in SSA representation.
  */
#[inline] pub fn dominance_frontier_map<G: GraphType>(
    g: *mut G, 
    source: Option<<G as GraphType>::NodeRef>) -> HashMap<G::NodeRef,HashSet<G::NodeRef>> 
{
    todo!();
    /*
        auto idomMap = immediateDominatorMap(g, source);
      std::unordered_map<
          typename G::NodeRef,
          std::unordered_set<typename G::NodeRef>>
          domFrontierMap;
      for (const auto node : g->getMutableNodes()) {
        if (node->getInEdges().size() < 2) {
          continue;
        }
        for (auto inEdge : node->getInEdges()) {
          auto predecessor = inEdge->tail();
          // This variable will track all the way up the dominator tree.
          auto runner = predecessor;
          while (runner != idomMap[node]) {
            domFrontierMap[runner].insert(node);
            runner = idomMap[runner];
          }
        }
      }
      return domFrontierMap;
    */
}

