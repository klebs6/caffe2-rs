crate::ix!();

/**
  | \brief Return a short string name for the
  | given \param node.
  |
  | The function works with both tensors and
  | operators.
  */
#[inline] pub fn get_node_name(node: NNGraph_NodeRef) -> String {
    
    todo!();
    /*
        if (!node) {
        return "";
      }
      if (nn::is<NeuralNetOperator>(node)) {
        if (auto* op = nn::get<NeuralNetOperator>(node)) {
          return op->getName();
        }
      }
      if (nn::is<NeuralNetData>(node)) {
        if (auto tensor = nn::get<NeuralNetData>(node)) {
          return "%" + tensor->getName();
        }
      }
      return "";
    */
}

