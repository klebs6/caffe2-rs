crate::ix!();

/**
  | The node has a single output and the output
  | has a single consumer.
  |
  */
#[inline] pub fn has_single_output_and_consumer<T,U>(node_ref: NodeRef<T,U>) -> bool {
    
    todo!();
    /*
    
    */
}

/// Create an output tensor node
///
#[inline] pub fn create_output<T,U>(
    nn:       *mut NNModule<T,U>,
    producer: NodeRef<T,U>,
    name:     String) -> NodeRef<T,U> {
    
    todo!();
    /*
    
    */
}

#[inline] pub fn get_outputs_from_node<T,U>(n: NodeRef<T,U>) -> Vec<NodeRef<T,U>> {
    
    todo!();
    /*
    
    */
}


#[inline] pub fn get_outputs_from_subgraph<T,U>(sg: &NNSubgraph) -> HashSet<NodeRef<T,U>> {
    
    todo!();
    /*
    
    */
}

#[inline] pub fn get_outputs<T,U>(n: NodeRef<T,U>) -> Vec<NodeRef<T,U>> {
    
    todo!();
    /*
        assert(
          is<NeuralNetOperator>(n) &&
          "getOutputs only works with NeuralNetOperator types.");
      std::vector<NodeRef> out;
      for (auto outEdge : n->getOutEdges()) {
        out.emplace_back(outEdge->head());
      }
      return out;
    */
}

#[inline] pub fn get_subgraph_outputs<T,U>(subgraph: &NNSubgraph) -> HashSet<NodeRef<T,U>> {
    
    todo!();
    /*
        std::set<NodeRef> subgraph_outputs;
      for (const auto& n : subgraph.getNodes()) {
        NOM_REQUIRE_OR_CONT(is<NeuralNetData>(n));
        if (hasConsumer(n)) {
          for (const auto& consumer : getConsumers(n)) {
            if (!subgraph.hasNode(consumer)) {
              subgraph_outputs.insert(n);
            }
          }
        } else {
          subgraph_outputs.insert(n);
        }
      }
      return subgraph_outputs;
    */
}

#[inline] pub fn create_output<T,U>(
    nn:       *mut NNModule<T,U>,
    producer: NodeRef<T,U>,
    name:     String) -> NodeRef<T,U> {
    
    todo!();
    /*
        auto outputNode =
          nn->dataFlow.createNode(std::make_unique<Tensor>(name));
      nn->dataFlow.createEdge(producer, outputNode);
      return outputNode;
    */
}
