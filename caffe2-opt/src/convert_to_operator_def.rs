crate::ix!();

#[inline] pub fn convert_to_operator_def<T,U>(instr_node: &NodeRef<T,U>) -> OperatorDef {
    
    todo!();
    /*
        auto* nnOp = repr::nn::get<repr::NeuralNetOperator>(instrNode);
      auto op_type = nnOp->getName();
      auto* annotation = nnOp->getAnnotation();
      caffe2::OperatorDef op;

      if (ConverterRegistry()->Has(op_type)) {
        op = ConverterRegistry()->Create(op_type)->convertToOperatorDef(nnOp);
      } else if (!annotation) {
        op.set_type(op_type);
      } else {
        if (isa<Caffe2Annotation>(annotation)) {
          auto c2_annotation = dyn_cast<Caffe2Annotation>(annotation);
          op = c2_annotation->getOperatorDef();
          op.mutable_device_option()->set_device_type(
              c2_annotation->getDeviceType());
        } else {
          CAFFE_THROW(
              "Couldn't convert operator annotation to Caffe2 operator def");
        }
      }

      // We may have swapped out some of the edges.
      op.clear_input();
      op.clear_output();
      return op;
    */
}

