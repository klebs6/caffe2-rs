crate::ix!();

impl StringJoinOp<CPUContext> {

    #[inline] pub fn do_run_with_type<T>(&mut self) -> bool {
    
        todo!();
        /*
            const auto& input = Input(0);

      CAFFE_ENFORCE_GT(input.numel(), 0);
      CAFFE_ENFORCE_LE(input.dim(), 2, "Only 1-D and 2-D tensors are supported");

      const auto* inputData = input.data<T>();
      int rowSize = (input.dim() == 2) ? input.size(1) : 1;
      if (this->axis_ == 0) {
        auto* output = Output(0, {input.size(0)}, at::dtype<std::string>());
        auto* outputData = output->template mutable_data<std::string>();

        int offset = 0;
        for (int i = 0; i < input.size(0); ++i) {
          std::stringstream stream;
          std::copy(
              inputData + offset,
              inputData + offset + rowSize,
              std::ostream_iterator<T>(stream, delimiter_.c_str()));
          outputData[i] = stream.str();
          offset += rowSize;
        }
      } else if (this->axis_ == 1) {
        auto* output = Output(0, {input.size(1)}, at::dtype<std::string>());
        auto* outputData = output->template mutable_data<std::string>();

        for (int j = 0; j < input.size(1); ++j) {
          std::stringstream stream;
          for (int i = 0; i < input.size(0); ++i) {
            stream << inputData[i * rowSize + j] << delimiter_;
          }
          outputData[j] = stream.str();
        }
      } else {
        CAFFE_ENFORCE(false, "Not supported");
      }

      return true;
        */
    }
}
