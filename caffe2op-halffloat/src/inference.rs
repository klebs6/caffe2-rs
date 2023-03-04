crate::ix!();

#[inline] pub fn float16_filler_tensor_inference(
    def: &OperatorDef,
    input: &Vec<TensorShape>) -> Vec<TensorShape> 
{
    todo!();
    /*
        vector<TensorShape> out(1);
      ArgumentHelper helper(def);
      out[0].set_data_type(static_cast<TensorProto_DataType>(
          helper.GetSingleArgument<int>("dtype", TensorProto_DataType_FLOAT16)));
      auto shape = helper.GetRepeatedArgument<int>("shape");
      for (int d : shape) {
        out[0].add_dims(d);
      }
      return out;
    */
}

#[inline] pub fn float_to_float16_ref(
    input:   *const f32,
    out:     *mut f16,
    n:       usize,
    do_clip: Option<bool>)  
{
    let do_clip: bool = do_clip.unwrap_or(false);

    todo!();
    /*
        if (do_clip) {
        constexpr float FP16_MAX = 65504.f;
        for (size_t i = 0; i < N; ++i) {
          out[i] = std::max(-FP16_MAX, std::min(in[i], FP16_MAX));
        }
      } else {
        for (size_t i = 0; i < N; ++i) {
          out[i] = in[i];
        }
      }
    */
}

#[inline] pub fn float_16to_float_ref(input: *const f16, out: *mut f32, n: usize)  {
    
    todo!();
    /*
        for (size_t i = 0; i < N; ++i) {
        out[i] = in[i];
      }
    */
}
