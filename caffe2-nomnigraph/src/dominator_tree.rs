crate::ix!();

/**
  | \brief A dominator tree finder.  Runs in
  | O(M*N), there exist more efficient
  | implementations.
  |
  | High level description of the algorithm:
  |
  | 1) Find a map of {node}->{dominator set}
  | --
  | allNodes = reachable(root)
  | for n in nodes:
  |   temporarily delete n from the graph
  |   dom[n] = allNodes - reachable(root)
  |   restore n to the graph
  |
  | 2) Construct tree from that map
  | --
  | starting at root, BFS in dominatorMap:
  |   if newnode has inedge, delete it
  |   draw edge from parent to child
  */
#[inline] pub fn dominator_tree<G: GraphType>(
    g:      *mut G, 
    source: Option<<G as GraphType>::NodeRef>) -> NomGraph<<G as GraphType>::NodeRef> 
{
    todo!();
    /*
        assert(
          g->getMutableNodes().size() > 0 &&
          "Cannot find dominator tree of empty graph.");
      if (!source) {
        auto rootSCC = tarjans(g).back();
        assert(
            rootSCC.getNodes().size() == 1 &&
            "Cannot determine source node topologically, please specify one.");
        for (auto& node : rootSCC.getNodes()) {
          source = node;
          break;
        }
      }

      Graph<typename G::NodeRef> tree;
      std::unordered_map<
          typename G::NodeRef,
          typename Graph<typename G::NodeRef>::NodeRef>
          mapToTreeNode;
      std::unordered_map<
          typename G::NodeRef,
          std::unordered_set<typename G::NodeRef>>
          dominatorMap;

      for (auto node : g->getMutableNodes()) {
        mapToTreeNode[node] = tree.createNode(std::move(node));
        if (node == source) {
          continue;
        }
        dominatorMap[source].insert(node);
      }

      for (const auto& node : g->getMutableNodes()) {
        if (node == source) {
          continue;
        }
        std::unordered_set<typename G::NodeRef> seen;
        std::unordered_set<typename G::NodeRef> dominated;
        reachable<G>(source, node, &seen);
        for (auto testNode : dominatorMap[source]) {
          if (seen.find(testNode) == seen.end() && testNode != node) {
            dominated.insert(testNode);
          }
        }
        dominatorMap[node] = dominated;
      }

      std::unordered_set<typename G::NodeRef> nextPass;
      nextPass.insert(source);

      while (nextPass.size()) {
        for (auto parent_iter = nextPass.begin(); parent_iter != nextPass.end();) {
          auto parent = *parent_iter;
          for (auto child : dominatorMap[parent]) {
            while (mapToTreeNode[child]->getInEdges().size()) {
              tree.deleteEdge(mapToTreeNode[child]->getInEdges().front());
            }
            tree.createEdge(mapToTreeNode[parent], mapToTreeNode[child]);
            if (dominatorMap.find(child) != dominatorMap.end()) {
              nextPass.insert(child);
            }
          }
          nextPass.erase(parent_iter++);
        }
      }

      return tree;
    */
}
