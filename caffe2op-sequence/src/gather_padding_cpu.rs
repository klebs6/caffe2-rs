crate::ix!();

impl GatherPaddingOp<CPUContext> {

    #[inline] pub fn gather_padding<T>(&mut self, 
        outer_size:        i32,
        lengths_size:      i32,
        block_size:        i32,
        pad_width:         i32,
        in_ptr:            *const T,
        lengths_ptr:       *const i32,
        padding_start_ptr: *mut T,
        padding_end_ptr:   *mut T)  {

        todo!();
        /*
            CAFFE_ENFORCE(
          (!std::is_same<bool, T>::value),
          "GatherPadding should not be executed on an input of type bool, as "
          "addition is not properly defined with booleans.");
      int64_t total_length = 0;
      for (int i = 0; i < lengths_size; ++i) {
        // check total length consistency
        const auto length = lengths_ptr[i];
        total_length += length;
        CAFFE_ENFORCE_LE(total_length, outer_size);
        // accumulate start paddings
        for (int j = 0; j < startPaddingWidth_; ++j) {
          for (int k = 0; k < block_size; ++k) {
            // Note: MSVC warns about unsafe use of type bool in operation.
            // This is now guarded by a CAFFE_ENFORCE so we can suppress it.
            #pragma warning(suppress: 4804)
            padding_start_ptr[k] += in_ptr[k];
          }
          in_ptr += block_size;
        }
        in_ptr += block_size * (length - pad_width);
        // accumulate end paddings
        for (int j = 0; j < endPaddingWidth_; ++j) {
          for (int k = 0; k < block_size; ++k) {
            #pragma warning(suppress: 4804)
            padding_end_ptr[k] += in_ptr[k];
          }
          in_ptr += block_size;
        }
      }
        */
    }
}
