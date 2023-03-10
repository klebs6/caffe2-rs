crate::ix!();

#[inline] pub fn cost_inference_for_concat(
    def:   &OperatorDef, 
    input: &Vec<TensorShape>) -> OpSchemaCost 
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
      vector<int> out_shape(in[0].dims().begin(), in[0].dims().end());
      if (add_axis) {
        out_shape.insert(out_shape.begin() + canonical_axis, in.size());
      } else {
        for (size_t i = 1; i < in.size(); ++i) {
          out_shape[canonical_axis] += in[i].dims(canonical_axis);
        }
      }
      uint64_t nElemRead = 1;
      for (int i = 0; i < in.size(); ++i) {
        nElemRead += nElemFromDim(in[i]);
      }
      int size = 1;
      for (auto& s : out_shape) {
        size *= s;
      }

      struct OpSchema::Cost cost;
      cost.flops = 0;
      cost.bytes_read = nElemRead * sizeof(in[0].data_type());
      cost.bytes_written = size * sizeof(in[0].data_type());
      cost.params_bytes = 0;
      return cost;
    */
}
