crate::ix!();

impl SummarizeOp<f32, CPUContext> {

    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& X = Input(0);
      const auto N = X.numel();
      CAFFE_ENFORCE_GT(N, 0);

      const float* Xdata = X.data<float>();
      double mean = 0;
      float max = Xdata[0];
      float min = Xdata[0];
      for (auto i = 0; i < N; ++i) {
        mean += static_cast<double>(Xdata[i]) / N;
        max = std::max(max, Xdata[i]);
        min = std::min(min, Xdata[i]);
      }
      // We will simply do a two-pass. More efficient solutions can be written but
      // I'll keep code simple for now.
      double standard_deviation = 0;
      for (auto i = 0; i < N; ++i) {
        double diff = Xdata[i] - mean;
        standard_deviation += diff * diff;
      }
      // Unbiased or biased? Let's do unbiased now.
      standard_deviation = N == 1 ? 0 : std::sqrt(standard_deviation / (N - 1));
      if (to_file_) {
        (*log_file_) << min << " " << max << " " << mean << " "
                     << standard_deviation << std::endl;
      }
      if (OutputSize()) {
        auto* Y = Output(0, {NUM_STATS}, at::dtype<float>());
        float* Ydata = Y->template mutable_data<float>();
        Ydata[MIN_IDX] = min;
        Ydata[MAX_IDX] = max;
        Ydata[MEAN_IDX] = static_cast<float>(mean);
        Ydata[STD_IDX] = static_cast<float>(standard_deviation);
      }
      return true;
        */
    }
}
