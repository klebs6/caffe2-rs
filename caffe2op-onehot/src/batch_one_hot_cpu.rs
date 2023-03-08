crate::ix!();

impl BatchOneHotOp<CPUContext> {
    
    #[inline] pub fn do_run_with_type<T>(&mut self) -> bool {
        todo!();
        /*
            auto& input = Input(X);
      auto& lens = Input(LENS);
      auto& vals = Input(VALS);
      CAFFE_ENFORCE_GE(input.dim(), 1);
      auto N = input.size(0);
      auto D = input.size_from_dim(1);
      CAFFE_ENFORCE_EQ(lens.numel(), D);

      const auto* lens_data = lens.template data<int32_t>();
      int64_t output_dim = 0;
      valsOffsets_.resize(D + 1);
      for (int64_t i = 0; i < D; i++) {
        CAFFE_ENFORCE_GE(lens_data[i], 0);
        valsOffsets_[i] = output_dim;
        output_dim += lens_data[i];
      }
      valsOffsets_[D] = output_dim;

      CAFFE_ENFORCE_EQ(vals.numel(), output_dim);

      auto* output = Output(ONE_HOT, {N, output_dim}, at::dtype<T>());

      const auto* input_data = input.template data<T>();
      const auto* vals_data = vals.template data<T>();
      auto* output_data = output->template mutable_data<T>();

      for (int64_t i = 0; i < N; ++i) {
        for (int64_t j = 0; j < D; j++) {
          const auto input_val = input_data[i * D + j];
          for (int64_t k = valsOffsets_[j]; k < valsOffsets_[j + 1]; ++k) {
            output_data[k] = vals_data[k] == input_val;
          }
        }
        output_data += output_dim;
      }

      return true;
        */
    }
}
