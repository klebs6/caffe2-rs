crate::ix!();

/// Get all nodes tracked by CF graph
#[inline] pub fn get_tracked_nodes<T,U>(
    cf: &mut NNCFGraph) -> HashSet<NodeRef<T,U>> {
    
    todo!();
    /*
        std::unordered_set<NodeRef> cfTrackedNodes;
      for (const auto& bbNode : cf.getMutableNodes()) {
        auto& bb = bbNode->data();
        for (const auto node : bb.getInstructions()) {
          cfTrackedNodes.insert(node);
        }
      }
      return cfTrackedNodes;
    */
}
