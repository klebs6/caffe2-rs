crate::ix!();

#[inline] pub fn tensor_inference_for_split(
    def:   &OperatorDef,
    input: &Vec<TensorShape>) -> Vec<TensorShape> 
{
    todo!();
    /*
        auto ret_invalid_shape = [&def]() {
        vector<TensorShape> out(def.output().size());
        for (auto& out_ts : out) {
          out_ts.set_unknown_shape(true);
        }
        return out;
      };
      // We only support shape inference of Split with 1 input
      if (def.input_size() != 1 || in.empty() || in.front().unknown_shape()) {
        return ret_invalid_shape();
      } else if (def.output_size() == 0) {
        return vector<TensorShape>();
      }
      ArgumentHelper helper(def);
      const int axis = helper.HasArgument("axis")
          ? helper.GetSingleArgument<int>("axis", -1)
          : GetDimFromOrderString(
                helper.GetSingleArgument<string>("order", "NCHW"));
      const int add_axis = helper.HasArgument("axis")
          ? helper.GetSingleArgument<int>("add_axis", 0)
          : 0;
      const auto& input = in[0];
      const int canonical_axis = canonical_axis_index_(axis, input.dims_size());
      const int input_channels = input.dims(canonical_axis);
      auto split = helper.GetRepeatedArgument<int>("split");
      // Equally split the input into outputs
      const int output_size = def.output_size();
      if (def.input_size() == caffe2::SplitOp<CPUContext>::kSplitOpInputSize) {
        if (!split.empty()) {
          LOG(WARNING) << "If you set split with an input blob, do not pass in "
                          "split in the argument.";
        }
        // We cannot infer output shape until we see the value of split input
        return ret_invalid_shape();
      } else if (split.empty()) {
        if (input_channels % output_size != 0) {
          LOG(WARNING) << "Input channels (" << input_channels
                       << ") should be divisible by number of outputs ("
                       << output_size << ")";
          return ret_invalid_shape();
        }
        split.resize(output_size, input_channels / output_size);
      } else if (split.size() != output_size) {
        LOG(WARNING) << "`split` size (" << split.size()
                     << ") should be equal to output size (" << output_size << ")";
        return ret_invalid_shape();
      }

      // Check validity of the split
      const int total_channels = add_axis
          ? def.output_size()
          : std::accumulate(split.begin(), split.begin() + output_size, 0);
      if (total_channels != input_channels) {
        LOG(WARNING) << "Input channels (" << input_channels
                     << ") is not equal to total output channels ("
                     << total_channels << ")";
        return ret_invalid_shape();
      }

      vector<int> output_dims(input.dims().begin(), input.dims().end());
      if (add_axis) {
        output_dims.erase(output_dims.begin() + canonical_axis);
      }
      vector<TensorShape> output_shapes;
      for (int i = 0; i < output_size; ++i) {
        if (!add_axis) {
          output_dims[canonical_axis] = split[i];
        }
        output_shapes.emplace_back(
            CreateTensorShape(output_dims, input.data_type()));
      }
      return output_shapes;
    */
}
