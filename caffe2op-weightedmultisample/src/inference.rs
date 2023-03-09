crate::ix!();

tensor_inference_function!{
    WeightedMultiSampling, 
    /* [](const OperatorDef& def,
                                const vector<TensorShape>& in) {
      vector<TensorShape> out(1);
      if (in[0].dims(0) == 0) {
        out[0].set_data_type(TensorProto::INT32);
        out[0].add_dims(0);
        return out;
      }

      const ArgumentHelper args(def);
      if (args.HasArgument("num_samples")) {
        CAFFE_ENFORCE_EQ(
            in.size(),
            1,
            "New shape must not be specified by the input blob and the "
            "argument `num_samples` at the same time.");
        int num_samples = args.GetSingleArgument<int64_t>("num_samples", 0);
        out[0] =
            CreateTensorShape(vector<int64_t>{num_samples}, TensorProto::INT32);
        return out;
      } else {
        CAFFE_ENFORCE_EQ(
            in.size(),
            2,
            "New shape must be specified by either the input blob or the "
            "argument `num_samples`.");
        std::vector<int64_t> output_dims = GetDimsVector(in[1]);
        out[0] = CreateTensorShape(output_dims, TensorProto::INT32);
        return out;
      }
    } */
}
