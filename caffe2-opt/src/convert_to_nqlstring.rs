crate::ix!();

/**
  | \brief Return a string representing the given
  | graph \param g.
  |
  | The returned string is a valid NQL query.
  */
#[inline] pub fn convert_to_nqlstring(g: &mut NNGraph) -> String {
    
    todo!();
    /*
        // Order nodes in a topological order.
      // TODO: Currently tarjans mutates the graph, and that's the only reason we
      // are not using const reference for `g`. We need to fix tarjans so that it
      // doesn't mutate the graph and use const reference in this function too.
      auto topoMatch = nom::algorithm::tarjans(&g);
      std::vector<NNGraph_NodeRef> nodes;
      int sccNum = 0;
      for (auto scc : topoMatch) {
        sccNum++;
        for (auto node : scc.getNodes()) {
          nodes.emplace_back(node);
        }
      }
      std::reverse(nodes.begin(), nodes.end());

      // Different nodes might have the same name. We want to change that so that
      // they are distinguishable by the name. NQL assumes that names are unique.
      std::unordered_map<NNGraph_NodeRef, std::string> renameMap =
          computeDedupRenameMap(nodes);

      // Going from top to bottom (nodes are in topological order), print all
      // nodes.
      std::string result = "def nn {\n";
      for (auto node : nodes) {
        std::string r = getNQLStringForBlob(node, renameMap);
        if (!r.empty()) {
          result += "  " + r + "\n";
        }
      }
      result += "}\n";
      return result;
    */
}
