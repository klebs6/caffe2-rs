crate::ix!();

/// \brief Create subgraph object from graph.
#[inline] pub fn create_subgraph<G: GraphType>(g: *mut G) -> <G as GraphType>::SubgraphType {

    todo!();
    /*
        typename GraphType::SubgraphType subgraph;
      for (auto& node : g->getMutableNodes()) {
        subgraph.addNode(node);
      }
      induceEdges(&subgraph);
      return subgraph;
    */
}

/// Create an operator
#[inline] pub fn create_operator<T, U, Args>(nn: *mut NNModule<T,U>, args: Args) -> NodeRef<T,U> {

    todo!();
    /*
        return nn->dataFlow.createNode(std::make_unique<T>(args));
    */
}
