crate::ix!();

impl RemovePaddingOp<CPUContext> {
    
    #[inline] pub fn do_run_with_type<T>(&mut self) -> bool {
    
        todo!();
        /*
            const auto& in = Input(0);
      CAFFE_ENFORCE_GE(in.dim(), 1);
      const int32_t outer_size = in.sizes()[0];
      const auto block_size = std::accumulate(
          in.sizes().begin() + 1, in.sizes().end(), 1, std::multiplies<int64_t>());
      const auto pad_width = startPaddingWidth_ + endPaddingWidth_;

      // if no lengths is provided, assume it is a single full-span entry
      const int32_t* lengths_ptr = &outer_size;
      int64_t lengths_size = 1;
      if (InputSize() > 1) {
        const auto& lengths = Input(1);
        lengths_ptr = lengths.data<int32_t>();
        lengths_size = lengths.numel();
      }

      auto out_dims = in.sizes().vec();
      out_dims[0] -= pad_width * lengths_size;
      auto* out = Output(0, std::move(out_dims), at::dtype<T>());

      const auto* in_ptr = in.template data<T>();
      auto* out_ptr = out->template mutable_data<T>();
      int64_t total_length = 0;
      for (int i = 0; i < lengths_size; ++i) {
        // check that total length is consistent
        const auto length = lengths_ptr[i];
        total_length += length;
        CAFFE_ENFORCE_LE(total_length, outer_size);
        std::copy(
            in_ptr + block_size * startPaddingWidth_,
            in_ptr + block_size * (length - endPaddingWidth_),
            out_ptr);
        in_ptr += block_size * length;
        out_ptr += block_size * (length - pad_width);
      }
      if (OutputSize() == 1) {
        return true;
      }

      auto* lengths_out = Output(1, {lengths_size}, at::dtype<int32_t>());
      std::transform(
          lengths_ptr,
          lengths_ptr + lengths_size,
          lengths_out->template mutable_data<int32_t>(),
          [pad_width](int32_t x) { return x - pad_width; });
      return true;
        */
    }
}
