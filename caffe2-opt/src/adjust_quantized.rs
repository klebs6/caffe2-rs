crate::ix!();

#[inline] pub fn adjust_quantized_offset_impl<T>(t: *mut Tensor, offset: u8)  {

    todo!();
    /*
        auto* data = t->mutable_data<T>();
      for (size_t i = 0; i < t->numel(); ++i) {
        data[i] -= offset;
      }
    */
}

#[inline] pub fn adjust_quantized_offset(t: *mut Tensor, offset: u8)  {
    
    todo!();
    /*
        if (t->template IsType<uint8_t>()) {
        adjustQuantizedOffsetImpl<uint8_t>(t, offset);
      }
    */
}
