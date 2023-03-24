crate::ix!();

#[inline] pub fn convert_to_c2net<T,U>(
    sub: &TransformSubgraph<T,U>,
    infos: &HashMap<NodeRef<T,U>,GroupAnnotation>) -> NetDef 
{
    todo!();
    /*
        caffe2::NetDef net;
      for (auto node : sub.nodes) {
        if (nn::is<NeuralNetOperator>(node)) {
          const auto* nn_op = nn::get<NeuralNetOperator>(node);
          assert(
              isa<Caffe2Annotation>(nn_op->getAnnotation()) &&
              "Cannot get caffe2 op from NNOp");
          const auto& op_def =
              dyn_cast<Caffe2Annotation>(nn_op->getAnnotation())->getOperatorDef();
          net.add_op()->CopyFrom(op_def);
        }
      }
      for (const auto kv : sub.external_input_refs) {
        net.add_external_input(kv.first);
        VLOG(2) << "Adding external input: " << kv.first;
      }
      for (const auto& kv : sub.external_output_refs) {
        net.add_external_output(kv.first);
        VLOG(2) << "Adding external output: " << kv.first;
      }

      return net;
    */
}
