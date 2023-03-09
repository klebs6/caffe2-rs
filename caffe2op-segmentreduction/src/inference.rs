crate::ix!();

#[inline] pub fn cost_inference_for_sparse_lengths(
    def:        &OperatorDef,
    inputs:     &Vec<TensorShape>,
    use_weight: bool) -> OpSchemaCost {
    
    todo!();
    /*
        int min_num_of_inputs = 3 + use_weight;
      CAFFE_ENFORCE_GE(
          inputs.size(),
          min_num_of_inputs,
          def.type() + " requires at least " + c10::to_string(min_num_of_inputs));

      const TensorShape data = inputs[0];
      const TensorShape indices = inputs[1 + use_weight];
      const TensorShape lengths = inputs[2 + use_weight];

      OpSchema::Cost c;
      CAFFE_ENFORCE_GT(data.dims_size(), 0, "data requires at least 1 dimension");
      uint64_t N = data.dims(0);
      if (N == 0) {
        return c;
      }
      uint64_t D = nElemFromDim(data, 1);
      CAFFE_ENFORCE_GT(
          lengths.dims_size(), 0, "lengths requires at least 1 dimension");
      uint64_t M = lengths.dims(0);
      uint64_t indices_size = nElemFromDim(indices);

      c.flops = indices_size * D;
      c.bytes_read = indices_size *
              (D * sizeof(data.data_type()) + sizeof(indices.data_type())) +
          M * sizeof(lengths.data_type());
      c.params_bytes = N * D * sizeof(data.data_type());
      if (use_weight) {
        const TensorShape weights = inputs[1];
        c.flops += indices_size * D;
        c.bytes_read += indices_size * sizeof(weights.data_type());
      }

      return c;
    */
}


