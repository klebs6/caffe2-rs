crate::ix!();

impl BucketizeOp<CPUContext> {

    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& input = Input(X);
      CAFFE_ENFORCE_GE(input.dim(), 1);

      auto N = input.numel();
      auto* output = Output(INDICES, input.sizes(), at::dtype<int32_t>());
      const auto* input_data = input.template data<float>();
      auto* output_data = output->template mutable_data<int32_t>();

      math::Set<int32_t, CPUContext>(output->numel(), 0.0, output_data, &context_);

      for (int64_t pos = 0; pos < N; pos++) {
        // here we assume the boundary values for each feature are sorted
        auto bucket_idx =
            std::lower_bound(
                boundaries_.begin(), boundaries_.end(), input_data[pos]) -
            boundaries_.begin();
        output_data[pos] = bucket_idx;
      }

      return true;
        */
    }
}

