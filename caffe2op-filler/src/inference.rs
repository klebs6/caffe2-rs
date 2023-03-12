crate::ix!();

//template <int VALUE_TYPE = TensorProto_DataType_FLOAT>
#[inline] pub fn filler_tensor_inference<const VALUE_TYPE: i32>(
    def:   &OperatorDef,
    input: &Vec<TensorShape>) -> Vec<TensorShape> 
{
    todo!();
    /*
        vector<TensorShape> out(1);
      ArgumentHelper helper(def);
      out[0].set_data_type(static_cast<TensorProto_DataType>(
          helper.GetSingleArgument<int>("dtype", VALUE_TYPE)));

      if (in.size()) {
        // TODO
        bool input_as_shape =
            helper.GetSingleArgument<bool>("input_as_shape", false);
        if (input_as_shape) {
          out[0].set_unknown_shape(true);
          return out;
        }
        for (auto d : in[0].dims()) {
          out[0].add_dims(d);
        }
      } else {
        auto shape = helper.GetRepeatedArgument<int64_t>("shape");
        for (auto d : shape) {
          out[0].add_dims(d);
        }
      }
      return out;
    */
}
