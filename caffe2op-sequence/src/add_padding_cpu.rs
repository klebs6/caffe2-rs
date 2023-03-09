crate::ix!();

impl AddPaddingOp<CPUContext> {

    #[inline] pub fn make_padding<T>(&mut self, 
        in_ptr:            *const T,
        out_ptr:           *mut T,
        lengths_ptr:       *const i32,
        lengths_size:      i32,
        outer_size:        i32,
        padding_start_ptr: *const T,
        padding_end_ptr:   *const T,
        block_size:        i64) -> bool {
    
        todo!();
        /*
            if (!lengths_ptr) {
        lengths_ptr = &outer_size;
      }

      int64_t total_length = 0;
      for (int i = 0; i < lengths_size; ++i) {
        // check that total length is consistent
        const auto length = lengths_ptr[i];
        total_length += length;
        CAFFE_ENFORCE_LE(total_length, outer_size);
        // copy padding before
        if (!padding_start_ptr) {
          memset(out_ptr, 0, block_size * startPaddingWidth_ * sizeof(T));
          out_ptr += block_size * startPaddingWidth_;
        } else {
          for (int j = 0; j < startPaddingWidth_; ++j) {
            std::copy(padding_start_ptr, padding_start_ptr + block_size, out_ptr);
            out_ptr += block_size;
          }
        }
        // copy payload
        const auto num_elems = block_size * length;
        std::copy(in_ptr, in_ptr + num_elems, out_ptr);
        in_ptr += num_elems;
        out_ptr += num_elems;
        // copy padding after
        if (!padding_end_ptr) {
          memset(out_ptr, 0, block_size * endPaddingWidth_ * sizeof(T));
          out_ptr += block_size * endPaddingWidth_;
        } else {
          for (int j = 0; j < endPaddingWidth_; ++j) {
            std::copy(padding_end_ptr, padding_end_ptr + block_size, out_ptr);
            out_ptr += block_size;
          }
        }
      }
      if (OutputSize() == 1) {
        return true;
      }

      auto* lengths_out = Output(1, {lengths_size}, at::dtype<int32_t>());
      const auto pad_width = startPaddingWidth_ + endPaddingWidth_;
      std::transform(
          lengths_ptr,
          lengths_ptr + lengths_size,
          lengths_out->template mutable_data<int32_t>(),
          [pad_width](int32_t x) { return x + pad_width; });
      return true;
        */
    }
}
