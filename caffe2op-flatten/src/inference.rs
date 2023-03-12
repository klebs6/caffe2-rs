crate::ix!();

#[inline] pub fn tensor_inference_for_flatten(
    def:   &OperatorDef,
    input: &Vec<TensorShape>) -> Vec<TensorShape> {
    
    todo!();
    /*
        ArgumentHelper helper(def);
      const int axis = helper.GetSingleArgument<int>("axis", 1);
      std::vector<TensorShape> out(1);
      int64_t outer = 1;
      int64_t inner = 1;
      std::size_t index = 0;
      for (auto d : in[0].dims()) {
        if (index < axis) {
          outer *= d;
        } else {
          inner *= d;
        }
        ++index;
      }
      out[0].set_data_type(in[0].data_type());
      out[0].add_dims(outer);
      out[0].add_dims(inner);
      return out;
    */
}
