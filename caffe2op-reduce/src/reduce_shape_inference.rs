crate::ix!();

#[inline] pub fn reduce_shape_inference(
    def:   &OperatorDef, 
    input: &Vec<TensorShape>) -> Vec<TensorShape> 
{
    
    todo!();
    /*
        if (in.size() != 1) {
        return std::vector<TensorShape>{
            CreateTensorShape({}, TensorProto_DataType_UNDEFINED)};
      }

      const auto& dims = in.front().dims();
      ArgumentHelper helper(def);
      std::vector<TensorShape> out;
      out.emplace_back();
      auto& ts = out.back();
      auto axis = helper.GetRepeatedArgument<int32_t>("axes");
      std::sort(axis.begin(), axis.end());
      auto keepdims = helper.GetSingleArgument<bool>("keepdims", true);
      size_t cursor = 0;
      int32_t id = 0;
      for (const auto d : dims) {
        if (cursor < axis.size() && id == axis[cursor]) {
          if (keepdims) {
            ts.add_dims(d == 0 ? 0 : 1);
          }
          ++cursor;
        } else {
          ts.add_dims(d);
        }
        ++id;
      }
      if (ts.dims_size() == 0 && dims.size() != 0) {
        ts.add_dims(1);
      }
      if (cursor != axis.size()) {
        ts.set_unknown_shape(true);
      }
      ts.set_data_type(in.front().data_type());
      return out;
    */
}
