crate::ix!();

impl WeightedSampleOp<f32, CPUContext> {

    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            CAFFE_ENFORCE_EQ(
          InputSize(),
          OutputSize(),
          "The number of tensors of the input and the output must be the same.");
      auto& weights = Input(0);
      int batch_size = weights.size(0);
      int weights_dim = weights.size(1);

      if (batch_size > 0 && weights_dim > 0) {
        cum_mass_.resize(weights_dim);
        const float* mat_weights = weights.template data<float>();
        const float* mat_values = nullptr;
        auto* out_idx = Output(0, {batch_size, 1}, at::dtype<int>());
        int* output_indices = out_idx->template mutable_data<int>();
        float* output_values = nullptr;

        if (InputSize() == 2) {
          auto& values = Input(1);
          CAFFE_ENFORCE_EQ(
              weights.sizes(),
              values.sizes(),
              "The sampling weights tensor and the sampling values tensor must have the same dimensions.");
          mat_values = values.template data<float>();

          auto* out_value = Output(1, {batch_size, 1}, at::dtype<float>());
          output_values = out_value->template mutable_data<float>();
        }

        for (int i = 0; i < batch_size; i++) {
          float r;
          int offset = i * weights_dim;

          cum_mass_[0] = mat_weights[offset];
          for (int j = 1; j < weights_dim; j++) {
            cum_mass_[j] = cum_mass_[j - 1] + mat_weights[offset + j];
          }

          math::RandUniform<float, CPUContext>(
              1, 0.0f, cum_mass_[cum_mass_.size() - 1], &r, &context_);
          // Makes the element in cum_mass_ slightly bigger
          // to compensate inaccuracy introduced due to rounding,
          cum_mass_[cum_mass_.size() - 1] += 0.01f;
          auto lb = lower_bound(cum_mass_.begin(), cum_mass_.end(), r);
          CAFFE_ENFORCE(lb != cum_mass_.end(), "Cannot find ", r, " in cum_mass_.");
          output_indices[i] = static_cast<int>(lb - cum_mass_.begin());

          if (output_values) {
            output_values[i] =
                static_cast<float>(mat_values[offset + (lb - cum_mass_.begin())]);
          }
        }
      } else {
        auto* out_idx = Output(0, {0}, at::dtype<int>());
        if (OutputSize() == 2) {
          auto* out_value = Output(1, {0}, at::dtype<float>());
          out_value->template mutable_data<float>();
        }
      }

      return true;
        */
    }
}
