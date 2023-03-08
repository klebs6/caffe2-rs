crate::ix!();

register_cpu_operator!{
    UnpackSegments, 
    UnpackSegmentsOp<CPUContext>
}

impl UnpackSegmentsOp<CPUContext> {

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
      auto* output = Output(0);

      CAFFE_ENFORCE_GE(data.dim(), 2, "DATA should be at least 2-D");
      CAFFE_ENFORCE_EQ(lengths.dim(), 1, "LENGTH should be 1-D");
      if (max_length_ != -1) {
        CAFFE_ENFORCE_EQ(
            max_length_,
            data.size(1),
            "max_length should be equal to the second dimension of the packed segments");
      }
      const T* l = lengths.template data<T>();

      int64_t total_l = 0;
      if (max_length_ != -1) {
        for (int64_t i = 0; i < lengths.size(0); ++i) {
          total_l += (int64_t)(l[i] <= max_length_ ? l[i] : max_length_);
        }
      } else {
        total_l = std::accumulate(l, l + lengths.size(0), (int64_t)0);
      }

      auto shape = data.sizes().vec();
      CAFFE_ENFORCE_EQ(
          shape[0], lengths.size(0), "LENGTH should match DATA in dimension 0");
      shape.erase(shape.begin());
      shape[0] = total_l;
      output->Resize(shape);
      // create output tensor
      auto* out = static_cast<char*>(output->raw_mutable_data(data.dtype()));
      if (!(data.size(0) && data.size(1))) {
        return true;
      }
      auto block_size = data.size_from_dim(2);
      auto block_bytesize = data.itemsize() * block_size;
      const auto* d = static_cast<const char*>(data.raw_data());
      int64_t start = 0;
      for (int64_t i = 0; i < lengths.size(0); ++i) {
        auto len = l[i];
        if (max_length_ != -1 && l[i] > max_length_) {
          len = max_length_;
        }
        context_.CopyItemsSameDevice(
            data.dtype(),
            len * block_size,
            d + block_bytesize * data.size(1) * i,
            out + block_bytesize * start);
        start += len;
      }
      return true;
        */
    }
}
