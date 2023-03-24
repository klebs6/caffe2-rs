crate::ix!();

#[inline] pub fn prepare_chain_graph_nodes(
    operator_nodes: &Vec<OperatorNode>, 
    execution_chains: &Vec<Vec<i32>>) -> Vec<OpGraphNode> 
{
    todo!();
    /*
        std::unordered_map<int, int> op_to_chain_idx;
      for (int chain_idx = 0; chain_idx < (int)execution_chains.size(); ++chain_idx) {
        const auto& chain_indices = execution_chains[chain_idx];
        for (const auto& chain_op_idx : chain_indices) {
          CAFFE_ENFORCE(!op_to_chain_idx.count(chain_op_idx));
          op_to_chain_idx[chain_op_idx] = chain_idx;
        }
      }

      std::vector<OpGraphNode> chain_nodes(execution_chains.size());
      for (int op_idx = 0; op_idx < (int)operator_nodes.size(); ++op_idx) {
        CAFFE_ENFORCE(op_to_chain_idx.count(op_idx));
        auto chain_idx = op_to_chain_idx[op_idx];
        auto& chain = chain_nodes[chain_idx];
        auto& op_node = operator_nodes[op_idx];

        for (const auto& child_idx : op_node.children_) {
          CAFFE_ENFORCE(op_to_chain_idx.count(child_idx));
          auto child_chain_idx = op_to_chain_idx[child_idx];
          if (child_chain_idx != chain_idx) {
            auto it = std::find(
                chain.children_.begin(), chain.children_.end(), child_chain_idx);
            if (it == chain.children_.end()) {
              chain.children_.push_back(child_chain_idx);
            }
          }
        }

        for (const auto& parent_idx : op_node.parents_) {
          CAFFE_ENFORCE(op_to_chain_idx.count(parent_idx));
          auto parent_chain_idx = op_to_chain_idx[parent_idx];
          if (parent_chain_idx != chain_idx) {
            auto it = std::find(
                chain.parents_.begin(), chain.parents_.end(), parent_chain_idx);
            if (it == chain.parents_.end()) {
              chain.parents_.push_back(parent_chain_idx);
            }
          }
        }
      }

      return chain_nodes;
    */
}
