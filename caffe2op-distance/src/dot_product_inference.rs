crate::ix!();

#[inline] pub fn tensor_inference_for_dot_product(
    def:   &OperatorDef,
    input: &Vec<TensorShape>) -> Vec<TensorShape> {
    
    todo!();
    /*
        CAFFE_ENFORCE_GT(in.size(), 0);

      vector<int64_t> dims(1);
      dims[0] = in[0].dims().size() > 0 ? in[0].dims(0) : 1;
      return vector<TensorShape>{CreateTensorShape(dims, in[0].data_type())};
    */
}

#[inline] pub fn cost_inference_for_dot_product(
    def:   &OperatorDef,
    input: &Vec<TensorShape>) -> OpSchemaCost {
    
    todo!();
    /*
        std::vector<TensorShape> out = TensorInferenceForDotProduct(def, in);
      CAFFE_ENFORCE_GT(out.size(), 0);
      CAFFE_ENFORCE_EQ(out[0].dims().size(), 1);

      struct OpSchema::Cost c = PointwiseCostInference<2>(def, in);
      c.bytes_written = out[0].dims(0) * sizeof(out[0].data_type());
      c.params_bytes = 0;
      return c;
    */
}

