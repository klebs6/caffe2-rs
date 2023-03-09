crate::ix!();

tensor_inference_function!{
    Transpose, 
    /* [](const OperatorDef& def,
                                const vector<TensorShape>& in) {
      ArgumentHelper helper(def);
      vector<int> axes = helper.GetRepeatedArgument<int>("axes");
      vector<TensorShape> out(1);
      out[0].set_data_type(in[0].data_type());

      if (axes.empty()) {
        for (auto axis = in [0].dims().rbegin(); axis != in[0].dims().rend();
             ++axis) {
          out[0].add_dims(*axis);
        }
      } else {
        auto tensor_size = in[0].dims().size();
        auto valid_axes =
            std::all_of(axes.begin(), axes.end(), [&tensor_size](int& axis) {
              return axis >= 0 && axis < tensor_size;
            });

        CAFFE_ENFORCE(valid_axes, "Axes argument passed in had invalid values");
        CAFFE_ENFORCE(
            axes.size() == tensor_size,
            "Axes argument passed in had the incorrect size");

        for (auto axis = axes.begin(); axis != axes.end(); ++axis) {
          out[0].add_dims(in[0].dims().Get(*axis));
        }
      }

      return out;
    } */
}
