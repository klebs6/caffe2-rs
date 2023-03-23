crate::ix!();

#[inline] pub fn replace_all_uses_with<T,U>(
    old_tensor_node: NodeRef<T,U>, 
    new_tensor_node: NodeRef<T,U>)  {
    
    todo!();
    /*
        const auto edges = oldTensorNode->getOutEdges();
      for (const auto& edge : edges) {
        edge->setTail(newTensorNode);
        oldTensorNode->removeOutEdge(edge);
        newTensorNode->addOutEdge(edge);
      }
    */
}

/**
  | Replace the producer of the first argument
  | with the second argument
  |
  */
#[inline] pub fn replace_producer<T,U>(tensor_node: NodeRef<T,U>, new_producer: NodeRef<T,U>)  {
    
    todo!();
    /*
    
    */
}

/**
  | Set all consumers of first argument
  | to consume the second argument
  |
  */
#[inline] pub fn replace_all_uses_with<T,U>(old_tensor_node: NodeRef<T,U>, new_tensor_node: NodeRef<T,U>)  {
    
    todo!();
    /*
    
    */
}

/**
  | Set the second argument to consume the
  | inputs of the first argument
  |
  */
#[inline] pub fn replace_as_consumer<T,U>(old_consumer: NodeRef<T,U>, new_consumer: NodeRef<T,U>)  {
    
    todo!();
    /*
    
    */
}
