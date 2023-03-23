crate::ix!();

#[inline] pub fn convert_node<NewT,OldT,T,U>(g: &mut NNGraph, node: NodeRef<T,U>) -> NodeRef<T,U> {

    todo!();
    /*
        assert(is<OldT>(node) && "Cannot get type from node.");

      NeuralNetOperator* nnOpPtr =
          dyn_cast<NeuralNetOperator>(node->mutableData()->release());

      auto newNode =
          g.createNode(std::make_unique<NewT>(*dyn_cast<OldT>(nnOpPtr)));

      g.replaceNode(node, newNode);
      g.deleteNode(node);

      return newNode;
    */
}
