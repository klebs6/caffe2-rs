crate::ix!();

impl WeightedSigmoidCrossEntropyWithLogitsOp<f32, CPUContext> {

    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& logits = Input(0);
      auto& targets = Input(1);
      auto& weights = Input(2);
      CAFFE_ENFORCE(logits.sizes() == targets.sizes());
      CAFFE_ENFORCE(weights.sizes() == targets.sizes());
      const auto inner_size = logits.dim() > 0 ? logits.sizes().back() : 1;
      const auto outer_size = logits.numel() / inner_size;

      std::vector<int64_t> dims;
      if (logits.dim() != 0) {
        dims =
            std::vector<int64_t>(logits.sizes().begin(), logits.sizes().end() - 1);
      }

      auto* out = Output(0, dims, at::dtype<float>());
      auto* out_ptr = out->template mutable_data<float>();

      auto* logits_ptr = logits.data<float>();
      auto* targets_ptr = targets.data<float>();
      auto* weights_ptr = weights.data<float>();

      auto in_idx = 0;
      for (int i = 0; i < outer_size; ++i) {
        float value = 0;
        for (int j = 0; j < inner_size; ++j) {
          value += sigmoid_xent_forward(logits_ptr[in_idx], targets_ptr[in_idx]) *
              weights_ptr[in_idx];
          ++in_idx;
        }
        out_ptr[i] = -value / inner_size;
      }
      return true;
        */
    }
}
