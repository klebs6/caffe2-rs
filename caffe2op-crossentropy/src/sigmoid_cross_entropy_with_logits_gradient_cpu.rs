crate::ix!();

impl SigmoidCrossEntropyWithLogitsGradientOp<f32, CPUContext> {

    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& g = Input(0);
      auto& logits = Input(1);
      auto& targets = Input(2);
      CAFFE_ENFORCE(logits.sizes() == targets.sizes());
      const auto inner_size = logits.dim() > 0 ? logits.sizes().back() : 1;
      const auto outer_size = logits.numel() / inner_size;
      CAFFE_ENFORCE(g.numel() == outer_size);

      auto* out = Output(0, logits.sizes(), at::dtype<float>());
      auto* out_ptr = out->template mutable_data<float>();

      auto* logits_ptr = logits.data<float>();
      auto* targets_ptr = targets.data<float>();
      auto* g_ptr = g.data<float>();

      auto in_idx = 0;
      for (int i = 0; i < outer_size; ++i) {
        auto g_factor = -g_ptr[i] / inner_size;
        for (int j = 0; j < inner_size; ++j) {
          if (unjoined_lr_loss_) {
            out_ptr[in_idx] = g_factor *
                unjoined_sigmoid_xent_backward(
                                  logits_ptr[in_idx], targets_ptr[in_idx]);
          } else {
            out_ptr[in_idx] = g_factor *
                (log_D_trick_ ? sigmoid_xent_backward_with_log_d_trick(
                                    logits_ptr[in_idx], targets_ptr[in_idx])
                              : sigmoid_xent_backward(
                                    logits_ptr[in_idx], targets_ptr[in_idx]));
          }
          ++in_idx;
        }
      }
      return true;
        */
    }
}
