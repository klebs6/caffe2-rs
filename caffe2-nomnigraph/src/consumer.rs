crate::ix!();

/**
  | The node has a unique consumer (there
  | may be multiple edges from output to
  | the single consumer).
  |
  */
#[inline] pub fn has_unique_consumer<T,U>(node_ref: NodeRef<T,U>) -> bool {
    
    todo!();
    /*
    
    */
}

#[inline] pub fn has_consumer<T,U>(n: NodeRef<T,U>) -> bool {
    
    todo!();
    /*
    
    */
}

#[inline] pub fn get_consumers<T,U>(n: NodeRef<T,U>) -> Vec<NodeRef<T,U>> {
    
    todo!();
    /*
    
    */
}

#[inline] pub fn has_consumer<T,U>(n: NodeRef<T,U>) -> bool {
    
    todo!();
    /*
        return n->getOutEdges().size() != 0;
    */
}

#[inline] pub fn get_consumers<T,U>(n: NodeRef<T,U>) -> Vec<NodeRef<T,U>> {
    
    todo!();
    /*
        assert(
          is<NeuralNetData>(n) &&
          "getProducer only works with NeuralNetData types.");
      std::vector<NodeRef> out;
      for (auto outEdge : n->getOutEdges()) {
        out.emplace_back(outEdge->head());
      }
      return out;
    */
}

#[inline] pub fn replace_as_consumer<T,U>(
    old_consumer: NodeRef<T,U>, 
    new_consumer: NodeRef<T,U>)
{
    todo!();
    /*
        const auto edges = oldConsumer->getInEdges();
      for (const auto& edge : edges) {
        edge->setHead(newConsumer);
        oldConsumer->removeInEdge(edge);
        newConsumer->addInEdge(edge);
      }
    */
}

#[inline] pub fn has_single_output_and_consumer<T,U>(node_ref: NodeRef<T,U>) -> bool {
    
    todo!();
    /*
        auto nodeOutputs = nn::getOutputs(nodeRef);
      NOM_REQUIRE_OR_RET_FALSE(nodeOutputs.size() == 1);
      auto nodeConsumers = nn::getConsumers(nodeOutputs.front());
      return nodeConsumers.size() == 1;
    */
}

#[inline] pub fn has_unique_consumer<T,U>(node_ref: NodeRef<T,U>) -> bool {
    
    todo!();
    /*
        auto nodeOutputs = nn::getOutputs(nodeRef);
      NodeRef nodeConsumer = nullptr;
      for (auto nodeOutput : nodeOutputs) {
        for (auto consumer : nn::getConsumers(nodeOutput)) {
          if (nodeConsumer && consumer && consumer != nodeConsumer) {
            return false;
          }
          nodeConsumer = consumer;
        }
      }
      return true;
    */
}
