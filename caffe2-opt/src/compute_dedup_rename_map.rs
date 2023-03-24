crate::ix!();

/**
  | Helper function for convertToNQLString
  | function.
  |
  | It takes a list of nodes and returns a map
  | node->unique_name. The new names are based on
  | the existing ones, but are also unique.
  */
#[inline] pub fn compute_dedup_rename_map(nodes: &Vec<NNGraph_NodeRef>) -> HashMap<NNGraph_NodeRef,String> {
    
    todo!();
    /*
        std::unordered_map<NNGraph_NodeRef, std::string> renameMap;
      std::unordered_set<std::string> takenNames;
      takenNames.clear();
      for (auto node : nodes) {
        std::string name = getNodeName(node);
        if (!isa<Data>(node->data())) {
          continue;
        }
        std::string newName = name;
        int dedupCounter = 0;
        while (takenNames.count(newName)) {
          newName = name + "_" + caffe2::to_string(dedupCounter);
          dedupCounter++;
        }
        renameMap[node] = newName;
        takenNames.insert(newName);
      }
      return renameMap;
    */
}
