crate::ix!();

#[inline] pub fn load_tensor_inference<const VALUE_TYPE: i32>(
    def:    &OperatorDef,
    unused: &Vec<TensorShape>) -> Vec<TensorShape> 
{
    todo!();
    /*
        ArgumentHelper helper(def);
      auto shape = helper.GetRepeatedArgument<int64_t>("shape");
      vector<TensorShape> out;
      // Currently load op supports only shape.
      // TODO: We have to extend it to support shapes vector.
      // Since it support just one shape, we return
      // the right shape information only when there is just one blob loaded.
      // Otherwise, we return unknown TensorShapes.
      if (def.output_size() == 1 && shape.size() > 0) {
        TensorShape ts;
        ts.set_data_type(static_cast<TensorProto_DataType>(
            helper.GetSingleArgument<int>("dtype", VALUE_TYPE)));
        for (auto d : shape) {
          ts.add_dims(d);
        }
        out.push_back(ts);
      } else {
        for (int i = 0; i < def.output_size(); i++) {
          TensorShape ts;
          ts.set_unknown_shape(true);
          out.push_back(ts);
        }
      }
      return out;
    */
}
