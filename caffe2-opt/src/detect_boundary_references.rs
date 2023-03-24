crate::ix!();

#[inline] pub fn detect_boundary_references<T,U>(
    subgraph:                 *mut TransformSubgraph<T,U>,
    infos:                    &HashMap<NodeRef<T,U>,GroupAnnotation>,
    original_external_output: &HashSet<String>)  
{
    
    todo!();
    /*
        for (auto node : subgraph->nodes) {
        // inputs
        for (auto in_edge : node->getInEdges()) {
          auto parent_node = in_edge->tail();
          const auto& info = GetInfo(infos, parent_node);
          if (info.group != subgraph->group_id &&
              nn::is<NeuralNetData>(parent_node)) {
            const auto* nn_tensor = nn::get<const NeuralNetData>(parent_node);
            subgraph->external_input_refs.emplace(
                nn_tensor->getName(), parent_node);
          }
        }

        // outputs
        if (!nn::is<NeuralNetData>(node)) {
          continue;
        }
        // Note that although matched subgraph won't contain external inputs as we
        // skip the initial input tensor of matching, it is possible to contain
        // external outputs. We will mark these external outputs as boundary outputs
        // too.
        auto name = nn::get<const NeuralNetData>(node)->getName();
        if (original_external_output.count(name)) {
          subgraph->external_output_refs.emplace(name, node);
        } else {
          for (auto child_node : nn::getConsumers(node)) {
            const auto& info = GetInfo(infos, child_node);
            if (info.group != subgraph->group_id) {
              subgraph->external_output_refs.emplace(name, node);
              break;
            }
          }
        }
      }
    */
}
