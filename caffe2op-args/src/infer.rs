crate::ix!();

#[inline] pub fn infer_tensor(
    def:   &OperatorDef,
    input: &Vec<TensorShape>) -> Vec<TensorShape> 
{
    todo!();
    /*
        std::vector<TensorShape> out(1);
      ArgumentHelper helper(def);
      int axis = helper.GetSingleArgument("axis", -1);
      const bool keep_dims = helper.GetSingleArgument("keepdims", true);
      const auto& in_dims = in[0].dims();
      auto* out_dims = out[0].mutable_dims();
      if (axis == -1) {
        axis = in_dims.size() - 1;
      }
      for (int i = 0; i < axis; ++i) {
        out_dims->Add(in_dims.Get(i));
      }
      if (keep_dims) {
        out_dims->Add(1);
      }
      for (int i = axis + 1; i < in_dims.size(); ++i) {
        out_dims->Add(in_dims.Get(i));
      }
      out[0].set_data_type(TensorProto::INT64);
      return out;
    */
}
