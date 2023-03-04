crate::ix!();

impl Float16UniformFillOp<CPUContext> {

    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto* output = Output(0, shape_, at::dtype<at::Half>());
      at::Half* out = output->template mutable_data<at::Half>();

      // Get a batch row by row and convert
      auto leading_dim_sz = output->size(0);
      int rowsz = output->numel() / output->size(0);

      vector<float> intermediate_data_;
      intermediate_data_.resize(rowsz);
      for (uint64_t i = 0; i < leading_dim_sz; i++) {
        math::RandUniform<float, CPUContext>(
            rowsz, min_, max_, intermediate_data_.data(), &context_);
        for (uint64_t j = 0; j < rowsz; j++) {
          out[i * rowsz + j] = intermediate_data_[j];
        }
      }
      return true;
        */
    }
}

