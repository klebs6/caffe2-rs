crate::ix!();

impl PercentileOp<CPUContext> {

    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            const auto& original_values = Input(X);
      CAFFE_ENFORCE_EQ(original_values.dim(), 2);
      const auto num_examples = original_values.size(0);
      const float* original_values_data = original_values.template data<float>();
      const auto num_features = original_values.size(1);

      const auto& value_pct_pairs = Input(VAL_PCT_PAIRS);
      CAFFE_ENFORCE_EQ(value_pct_pairs.dim(), 2);
      CAFFE_ENFORCE_EQ(value_pct_pairs.size(1), 2);
      const int num_values = value_pct_pairs.size(0);
      const float* value_pct_data = value_pct_pairs.template data<float>();

      const auto& lengths = Input(LENS);
      const int* lengths_data = lengths.template data<int>();
      CAFFE_ENFORCE_EQ(lengths.numel(), num_features);

      CAFFE_ENFORCE_EQ(
          std::accumulate(lengths_data, lengths_data + lengths.numel(), 0),
          num_values,
          "Sum of lengths should be equal to the total number of samples");

      ReinitializeTensor(
          &values_tensor,
          {num_values},
          at::dtype<float>().device(CPU));
      ReinitializeTensor(
          &percentiles_tensor,
          {num_values},
          at::dtype<float>().device(CPU));
      float* values_tensor_data = values_tensor.template mutable_data<float>();
      float* percentiles_tensor_data =
          percentiles_tensor.template mutable_data<float>();
      for (int ind = 0; ind < num_values; ind++) {
        values_tensor_data[ind] = value_pct_data[2 * ind];
        percentiles_tensor_data[ind] = value_pct_data[2 * ind + 1];
      }

      auto* percentile_values =
          Output(PCT, original_values.sizes(), at::dtype<float>());
      float* percentile_values_data =
          percentile_values->template mutable_data<float>();

      int current_ind = 0;
      int current_dist_start = 0;
      int current_length;
      for (int i = 0; i < num_examples; i++) {
        current_dist_start = 0;

        for (int j = 0; j < num_features; j++) {
          current_length = lengths_data[j];
          const auto lower_bound =
              std::lower_bound(
                  values_tensor_data + current_dist_start,
                  values_tensor_data + current_dist_start + current_length,
                  original_values_data[current_ind]) -
              values_tensor_data;
          if (lower_bound == current_dist_start + current_length) {
            percentile_values_data[current_ind] = 1.0;
          } else if (
              original_values_data[current_ind] ==
              values_tensor_data[lower_bound]) {
            percentile_values_data[current_ind] =
                percentiles_tensor_data[lower_bound];
          } else if (lower_bound == current_dist_start) {
            percentile_values_data[current_ind] = 0.0;
          } else {
            float lower_pct = percentiles_tensor_data[lower_bound - 1];
            float upper_pct = percentiles_tensor_data[lower_bound];
            float interval_length = values_tensor_data[lower_bound] -
                values_tensor_data[lower_bound - 1];
            float normalized_dist_to_lower = (original_values_data[current_ind] -
                                              values_tensor_data[lower_bound - 1]) /
                interval_length;
            percentile_values_data[current_ind] =
                lower_pct + normalized_dist_to_lower * (upper_pct - lower_pct);
          }
          current_dist_start += current_length;
          current_ind++;
        }
      }
      return true;
        */
    }
}
