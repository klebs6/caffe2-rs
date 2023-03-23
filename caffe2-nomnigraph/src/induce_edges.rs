crate::ix!();

/**
  | -----------
  | @brief
  | 
  | Induces edges on a subgraph by connecting
  | all nodes that are connected in the original
  | graph.
  |
  */
#[inline] pub fn induce_edges<SubgraphType>(sg: *mut SubgraphType)  {

    todo!();
    /*
        for (auto& node : sg->getNodes()) {
        // We can scan only the inEdges
        for (auto& inEdge : node->getInEdges()) {
          if (sg->hasNode(inEdge->tail())) {
            sg->addEdge(inEdge);
          }
        }
      }
    */
}

