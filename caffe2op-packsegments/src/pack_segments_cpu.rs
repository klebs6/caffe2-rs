crate::ix!();

register_cpu_operator!{
    PackSegments,   
    PackSegmentsOp<CPUContext>
}

impl PackSegmentsOp<CPUContext> {

    #[inline] pub fn do_run_with_type<T>(&mut self) -> bool {
        todo!();
        /*
            return DispatchHelper<
          TensorTypes2<char, int32_t, int64_t, float, std::string>,
          T>::call(this, Input(DATA));
        */
    }
    
    #[inline] pub fn do_run_with_type2<T, Data_T>(&mut self) -> bool {
        todo!();
        /*
            const auto& data = Input(DATA);
      const auto& lengths = Input(LENGTHS);

      Tensor* presence_mask = nullptr;
      if (return_presence_mask_) {
        presence_mask = Output(1);
      }

      CAFFE_ENFORCE_GE(data.dim(), 1, "DATA should be at least 1-D");
      CAFFE_ENFORCE_EQ(lengths.dim(), 1, "LENGTH should be 1-D");

      // Find the length of the longest sequence.
      const T* l = lengths.template data<T>();
      T max_length = 0;
      int64_t total_length = 0;
      for (T i = 0; i < lengths.size(0); ++i) {
        max_length = std::max(max_length, l[i]);
        total_length += l[i];
      }
      if (max_length_ != -1) {
        max_length = max_length_;
      }

      // Total lengths must be the same as data.dims(0)
      CAFFE_ENFORCE_EQ(
          data.size(0),
          total_length,
          " PackSegments requires that the sum of the lengths ",
          total_length,
          " is equal to the first data dimension ",
          data.size(0));

      auto shape =
          data.sizes().vec(); // Shape of output is batch_size x max_len x ...
      shape[0] = max_length;
      shape.insert(shape.begin(), lengths.numel());
      auto* output = Output(0, shape, at::dtype(data.dtype()));

      // create output tensor
      auto* out = static_cast<char*>(output->raw_mutable_data(data.dtype()));

      bool* presence_mask_data = nullptr;
      if (return_presence_mask_) {
        // Shape of presence is batch_size x max_len
        std::vector<int64_t> presence_shape{lengths.numel(), max_length};
        presence_mask->Resize(presence_shape);
        presence_mask_data = presence_mask->template mutable_data<bool>();
      }

      if (!data.size(0)) {
        // Return empty output (with the proper shape)
        return true;
      }

      // Do padding
      // Ignore string since math::Set does not support string.
      // For all other cases, the behavior should mimic the GPU version where the
      // padding is always zero for types other than float.
      // TODO(xinyizhang): potentially restructure to clean up the logic here.
      if (output->template IsType<float>()) {
        math::Set<float, CPUContext>(
            output->numel(),
            padding_,
            output->template mutable_data<float>(),
            &context_);
      } else if (output->template IsType<int32_t>()) {
        math::Set<int32_t, CPUContext>(
            output->numel(),
            0,
            output->template mutable_data<int32_t>(),
            &context_);
      } else if (output->template IsType<int64_t>()) {
        math::Set<int64_t, CPUContext>(
            output->numel(),
            0,
            output->template mutable_data<int64_t>(),
            &context_);
      } else if (output->template IsType<char>()) {
        math::Set<char, CPUContext>(
            output->numel(), 0, output->template mutable_data<char>(), &context_);
      }
      if (return_presence_mask_) {
        memset(presence_mask_data, (int)false, presence_mask->numel());
      }

      auto block_size = data.size_from_dim(1);
      auto block_bytesize = data.itemsize() * block_size;
      const auto* d = static_cast<const char*>(data.raw_data());
      int64_t start = 0;
      for (int64_t i = 0; i < lengths.size(0); ++i) {
        auto len = l[i] <= max_length ? l[i] : max_length;
        context_.CopyItemsSameDevice(
            data.dtype(),
            len * block_size,
            d + block_bytesize * start,
            out + block_bytesize * max_length * i);
        if (return_presence_mask_) {
          memset(presence_mask_data + max_length * i, (int)true, len);
        }
        start += l[i];
      }

      return true;
        */
    }
}
