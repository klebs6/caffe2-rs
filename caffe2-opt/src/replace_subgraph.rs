crate::ix!();

#[inline] pub fn replace_subgraph<T,U>(
    subgraph: &TransformSubgraph<T,U>,
    net_opt:  &mut NetDef,
    g:        *mut NNGraph)  
{

    todo!();
    /*
        // Delete the old subgraph starting from the input nodes until we hit boundary
      // tensors
      for (auto node : subgraph.nodes) {
        if (nn::is<NeuralNetData>(node) &&
            subgraph.external_output_refs.count(
                nn::get<const NeuralNetData>(node)->getName())) {
          VLOG(2) << "Keeping " << ShowNode(node);
          continue;
        }
        VLOG(2) << "Deleting " << ShowNode(node);
        g->deleteNode(node);
      }

      // Convert new NetDef back to NNGraph
      std::unordered_map<std::string, NodeRef> tensor_map;
      for (const auto kv : subgraph.external_input_refs) {
        tensor_map.emplace(kv.first, kv.second);
      }
      for (const auto kv : subgraph.external_output_refs) {
        tensor_map.emplace(kv.first, kv.second);
      }
      for (auto& op : *net_opt.mutable_op()) {
        auto op_node = g->createNode();
        for (const auto& input : op.input()) {
          if (!tensor_map.count(input)) {
            tensor_map[input] =
                g->createNode(std::make_unique<nom::repr::Tensor>(input));
          }

          auto tensor_node = tensor_map[input];
          g->createEdge(tensor_node, op_node);
        }

        for (const auto& output : op.output()) {
          if (!tensor_map.count(output)) {
            tensor_map[output] =
                g->createNode(std::make_unique<nom::repr::Tensor>(output));
          }
          auto tensor_node = tensor_map[output];
          g->createEdge(op_node, tensor_node);
        }

        op_node->resetData(convertToNeuralNetOperator(op));
      }
    */
}
