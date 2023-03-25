crate::ix!();

/// NeuralNetData specific helpers.
///
#[inline] pub fn has_producer<T,U>(n: NodeRef<T,U>) -> bool {
    
    todo!();
    /*
        return n->getInEdges().size() != 0;
    */
}

#[inline] pub fn get_producer<T,U>(n: NodeRef<T,U>) -> NodeRef<T,U> {
    
    todo!();
    /*
        assert(
          is<NeuralNetData>(n) &&
          "getProducer only works with NeuralNetData types.");
      auto inEdges = n->getInEdges();
      assert(inEdges.size() > 0 && "Tensor does not have a producer.");
      assert(
          inEdges.size() == 1 &&
          "Malformed NNGraph, NeuralNetData has multiple producers.");
      return inEdges.front()->tail();
    */
}

#[inline] pub fn replace_producer<T,U>(
    tensor_node: NodeRef<T,U>, 
    new_producer: NodeRef<T,U>) 
{
    
    todo!();
    /*
        assert(
          is<NeuralNetData>(tensorNode) &&
          "First argument must contain NeuralNetData");
      auto inEdges = tensorNode->getInEdges();
      assert(
          inEdges.size() == 1 && "Tensor node passed in does not have a producer");
      auto edge = inEdges.at(0);
      auto prevProducer = edge->tail();
      prevProducer->removeOutEdge(edge);
      edge->setTail(newProducer);
      newProducer->addOutEdge(edge);
    */
}
