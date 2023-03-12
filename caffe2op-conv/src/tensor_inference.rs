crate::ix!();

#[inline] pub fn tensor_inference_for_conv_gradient(
    def:   &OperatorDef,
    input: &Vec<TensorShape>) -> Vec<TensorShape> {
    
    todo!();
    /*
        CAFFE_ENFORCE_EQ(in.size(), 3U, "ConvGradient requires 3 inputs");

      if (in[0].unknown_shape()) {
        std::vector<TensorShape> out(1);
        out[0].set_unknown_shape(true);
        return out;
      }
      ArgumentHelper helper(def);
      const auto no_bias = helper.GetSingleArgument<int>("no_bias", 0);
      const auto n_outputs = def.output_size();
      vector<TensorShape> out(n_outputs);

      // FILTER_GRAD has the same shape as FILTER
      out[0] = in[1];
      if (!no_bias) {
        vector<int64_t> bias_shape = {in[1].dims(0)};
        out[1] = CreateTensorShape(bias_shape, in[1].data_type());
      }

      if (n_outputs == 3 || (no_bias && n_outputs == 2)) {
        // INPUT_GRAD has the same shape as INPUT
        out[out.size() - 1] = in[0];
      }

      return out;
    */
}
