crate::ix!();

impl BatchBucketOneHotOp<CPUContext> {

    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& input = Input(X);
      auto& lens = Input(LENS);
      auto& boundaries = Input(BOUNDARIES);
      CAFFE_ENFORCE_GE(input.dim(), 1);
      auto N = input.size(0);
      auto D = input.size_from_dim(1);
      CAFFE_ENFORCE_EQ(lens.numel(), D);

      const auto* lens_data = lens.template data<int32_t>();

      CAFFE_ENFORCE_EQ(
          std::accumulate(lens_data, lens_data + lens.numel(), 0),
          boundaries.numel(),
          "The sum of length should be equal to the length of boundaries");

      int64_t output_dim = 0;
      for (int64_t i = 0; i < D; i++) {
        CAFFE_ENFORCE_GT(lens_data[i], 0);
        // Number of buckets is number of bucket edges + 1
        output_dim += (lens_data[i] + 1);
      }

      auto* output = Output(ONE_HOT, {N, output_dim}, at::dtype<float>());

      const auto* input_data = input.template data<float>();
      const auto* boundaries_data = boundaries.template data<float>();
      auto* output_data = output->template mutable_data<float>();

      math::Set<float, CPUContext>(output->numel(), 0.f, output_data, &context_);

      int64_t pos = 0;
      for (int64_t i = 0; i < N; i++) {
        auto* boundaries_offset = boundaries_data;
        int64_t output_offset = 0;

        for (int64_t j = 0; j < D; j++) {
          // here we assume the boundary values for each feature are sorted
          int64_t lower_bucket_idx = std::lower_bound(
                                        boundaries_offset,
                                        boundaries_offset + lens_data[j],
                                        input_data[pos]) -
              boundaries_offset;

          int64_t upper_bucket_idx = std::upper_bound(
                                        boundaries_offset,
                                        boundaries_offset + lens_data[j],
                                        input_data[pos]) -
              boundaries_offset;

          int64_t bucket_idx = (lower_bucket_idx + upper_bucket_idx) / 2;
          output_data[i * output_dim + output_offset + bucket_idx] = 1.0;
          boundaries_offset += lens_data[j];
          output_offset += (lens_data[j] + 1);
          pos++;
        }
      }

      return true;
        */
    }
}
