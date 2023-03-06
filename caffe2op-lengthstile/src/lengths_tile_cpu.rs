crate::ix!();

impl LengthsTileOp<CPUContext> {

    #[inline] pub fn run_on_cpu_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& data = Input(DATA);
      auto& lengths = Input(LENGTHS);
      auto* output = Output(0);

      CAFFE_ENFORCE_EQ(lengths.dim(), 1, "LENGTHS must be 1-D");
      CAFFE_ENFORCE_GE(data.dim(), 1, "DATA should be at least 1-D");
      CAFFE_ENFORCE_EQ(lengths.numel(), data.size(0));

      // Context::CopyFrom and math::Sum need the same context to avoid race
      // conditions
      // why? CPUContext is not used in Sum
      lengths_host_.CopyFrom(lengths); // sync copy
      auto lengths_size = lengths_host_.numel();
      auto* lengths_data = lengths_host_.data<int32_t>();

      int32_t total_length = 0;
      CPUContext cpuContext;
      math::Sum<int32_t, CPUContext>(
          lengths_size, lengths_data, &total_length, &cpuContext);

      auto shape = data.sizes().vec();
      shape[0] = total_length;
      output->Resize(shape);

      auto block_bytesize = data.size_from_dim(1) * data.dtype().itemsize();
      auto src = static_cast<const char*>(data.raw_data());
      auto out = static_cast<char*>(output->raw_mutable_data(data.dtype()));

      for (int64_t i = 0; i < lengths_size; ++i) {
        auto length = lengths_data[i];
        CAFFE_ENFORCE_GE(length, 0);
        for (int32_t j = 0; j < length; ++j) {
          context_.CopyBytesSameDevice(block_bytesize, src, out);
          out += block_bytesize;
        }
        src += block_bytesize;
      }
      return true;
        */
    }
}
