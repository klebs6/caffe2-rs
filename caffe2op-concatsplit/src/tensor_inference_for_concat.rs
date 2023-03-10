crate::ix!();

#[inline] pub fn tensor_inference_for_concat(
    def:   &OperatorDef,
    input: &Vec<TensorShape>) -> Vec<TensorShape> 
{
    todo!();
    /*
        ArgumentHelper helper(def);
      const int axis = helper.HasArgument("axis")
          ? helper.GetSingleArgument<int>("axis", -1)
          : GetDimFromOrderString(
                helper.GetSingleArgument<string>("order", "NCHW"));
      bool add_axis = helper.GetSingleArgument<int>("add_axis", 0) != 0;
      int adj_size = in[0].dims_size() + (add_axis ? 1 : 0);
      const int canonical_axis = canonical_axis_index_(axis, adj_size);
      CAFFE_ENFORCE_LT(canonical_axis, adj_size, "Axis not in input ndim range.");
      CAFFE_ENFORCE_GT(in.size(), 0);
      vector<int> split_shape(1, in.size());
      vector<int> out_shape(in[0].dims().begin(), in[0].dims().end());
      if (add_axis) {
        for (int i = 1; i < in.size(); ++i) {
          CAFFE_ENFORCE_EQ(
              in[0].dims().size(),
              in[i].dims().size(),
              "All inputs of Concat should have same dims when add_axis = 1. "
              "Got different sizes for inputs 0 and ",
              i);
          for (int j = 0; j < in[0].dims().size(); ++j) {
            CAFFE_ENFORCE_EQ(
                in[0].dims(j),
                in[i].dims(j),
                "All inputs of Concat should have same dims when add_axis = 1. "
                "Got different dims for inputs 0 and ",
                i,
                ". At dim: ",
                j);
          }
        }
        out_shape.insert(out_shape.begin() + canonical_axis, in.size());
      } else {
        for (int i = 1; i < in.size(); ++i) {
          CAFFE_ENFORCE(
              in[0].dims_size() == in[i].dims_size() ||
                  (canonical_axis == in[0].dims_size() - 1 &&
                   in[0].dims_size() == in[i].dims_size() + 1),
              "All inputs of Concat should have same dims except "
              "canonical_axis dim that is equal to ",
              canonical_axis,
              "Got different sizes for inputs 0 and ",
              i);
          for (int j = 0; j < in[0].dims_size(); ++j) {
            if (j == canonical_axis) {
              continue;
            }
            CAFFE_ENFORCE_EQ(
                in[0].dims(j),
                in[i].dims(j),
                "All inputs of Concat should have same dims except "
                "canonical_axis dim that is equal to ",
                canonical_axis,
                "Got different dims for inputs 0 and ",
                i,
                ". At dim: ",
                j);
          }
        }

        for (int i = 1; i < in.size(); ++i) {
          out_shape[canonical_axis] += in[i].dims(canonical_axis);
        }
      }
      if (def.output_size() == 1) {
        return vector<TensorShape>{CreateTensorShape(out_shape, in[0].data_type())};
      }
      return vector<TensorShape>{
          CreateTensorShape(out_shape, in[0].data_type()),
          CreateTensorShape(split_shape, TensorProto::INT32)};
    */
}
