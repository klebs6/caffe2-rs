crate::ix!();

#[inline] pub fn cost_inference_for_conv_gradient(
    def:    &OperatorDef,
    inputs: &Vec<TensorShape>) -> OpSchemaCost 
{
    todo!();
    /*
        CAFFE_ENFORCE_EQ(inputs.size(), 3U, "ConvGradient requires 3 inputs");
      ArgumentHelper helper(def);
      const auto order =
          StringToStorageOrder(helper.GetSingleArgument<string>("order", "NCHW"));
      const auto no_bias = helper.GetSingleArgument<int>("no_bias", 0);
      const auto n_outputs = def.output_size();

      const auto& outputs = TensorInferenceForConvGradient(def, inputs);
      const auto& X = inputs[0];
      const auto& filter = inputs[1];
      const auto& dY = inputs[2];
      const auto N = X.dims(0);
      const auto M = filter.dims(0);
      const auto C =
          (order == StorageOrder::NCHW ? X.dims(1) : X.dims(X.dims_size() - 1));
      const auto output_image_size =
          (order == StorageOrder::NCHW
               ? nElemFromDim(dY, 2)
               : nElemBetweenDim(dY, 1, dY.dims_size() - 1));
      auto kernel_elem =
          (order == StorageOrder::NCHW
               ? nElemFromDim(filter, 2)
               : nElemBetweenDim(filter, 1, filter.dims_size() - 1));

      struct OpSchema::Cost c;
      c.flops = N * 2 * M * kernel_elem * C * output_image_size;
      if (!no_bias) {
        c.flops += N * (M * output_image_size);
      }
      if (n_outputs == 3 || (no_bias && n_outputs == 2)) {
        c.flops += N * 2 * M * kernel_elem * C * output_image_size;
      }

      c.bytes_read = (nElemFromDim(X) + nElemFromDim(filter) + nElemFromDim(dY)) *
          sizeof(float);

      for (auto i = 0; i < n_outputs; i++) {
        c.bytes_written += nElemFromDim(outputs[i]) * sizeof(float);
      }
      c.params_bytes = nElemFromDim(filter) * sizeof(float);

      return c;
    */
}
