crate::ix!();

#[inline] pub fn get_inputs_from_subgraph<T,U>(sg: &NNSubgraph) -> HashSet<NodeRef<T,U>> {
    
    todo!();
    /*
    
    */
}

#[inline] pub fn get_inputs_from_node<T,U>(n: NodeRef<T,U>) -> Vec<NodeRef<T,U>> {
    
    todo!();
    /*
    
    */
}

#[inline] pub fn has_inputs<T,U>(n: NodeRef<T,U>) -> bool {
    
    todo!();
    /*
        return n->getInEdges().size() != 0;
    */
}

#[inline] pub fn get_inputs<T,U>(n: NodeRef<T,U>) -> Vec<NodeRef<T,U>> {
    
    todo!();
    /*
        assert(
          is<NeuralNetOperator>(n) &&
          "getInputs only works with NeuralNetOperator types.");
      std::vector<NodeRef> out;
      for (auto inEdge : n->getInEdges()) {
        out.emplace_back(inEdge->tail());
      }
      return out;
    */
}

#[inline] pub fn get_subgraph_inputs<T,U>(subgraph: &NNSubgraph) -> HashSet<NodeRef<T,U>> {
    
    todo!();
    /*
        std::set<NodeRef> subgraph_inputs;
      for (const auto& node : subgraph.getNodes()) {
        NOM_REQUIRE_OR_CONT(is<NeuralNetData>(node));
        if (hasProducer(node)) {
          if (!subgraph.hasNode(getProducer(node))) {
            subgraph_inputs.insert(node);
          }
        } else {
          subgraph_inputs.insert(node);
        }
      }
      return subgraph_inputs;
    */
}
