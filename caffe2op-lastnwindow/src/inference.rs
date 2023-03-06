crate::ix!();

tensor_inference_function!{LastNWindowCollector, /* [](const OperatorDef& def,
                                const vector<TensorShape>& in) {
      auto output_size = def.output_size();
      vector<TensorShape> out(output_size);
      const ArgumentHelper helper(def);
      const auto num_to_collect =
          helper.GetSingleArgument<int>("num_to_collect", -1);

      const auto data_dims = GetDimsVector(in[2]);
      vector<int64_t> last_n_shape(data_dims.size());
      last_n_shape[0] = num_to_collect;
      std::copy(data_dims.begin() + 1, data_dims.end(), last_n_shape.begin() + 1);
      out[0] = CreateTensorShape(last_n_shape, in[2].data_type());

      out[1] = in[1];

      if (output_size > 2) {
        vector<int64_t> num_visited_shape(1);
        num_visited_shape[0] = 1;
        out[2] = CreateTensorShape(num_visited_shape, TensorProto::INT64);
      }

      return out;
    })*/
}
