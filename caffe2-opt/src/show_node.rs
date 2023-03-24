crate::ix!();

#[inline] pub fn show_node<T,U>(node: NodeRef<T,U>) -> String {
    
    todo!();
    /*
        if (nn::is<NeuralNetData>(node)) {
        const auto* nn_tensor = nn::get<NeuralNetData>(node);
        return c10::str("Tensor: ", nn_tensor->getName());
      } else if (nn::is<NeuralNetOperator>(node)) {
        const auto* nn_op = nn::get<NeuralNetOperator>(node);
        const auto& op_def =
            dyn_cast<Caffe2Annotation>(nn_op->getAnnotation())->getOperatorDef();
        return c10::str("Op: ", op_def.type());
      } else {
        CAFFE_THROW("Known node");
      }
    */
}
