crate::ix!();

impl DropoutOp<f32, CPUContext> {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& X = Input(0);
      auto* Y = Output(0, X.sizes(), at::dtype<float>());

      if (is_test_) {
        if (!IsInputOutputAlias(0, 0)) {
          context_.CopyFromCPU<float>(
              X.numel(), X.data<float>(), Y->template mutable_data<float>());
        }
        return true;
      } else {
        float scale = 1. / (1. - ratio_);
        // mask=true means keep, and mask=false means not keep, so we will
        // generate probability depending on 1-ratio.
        at::bernoulli_distribution<double> dist(1. - ratio_);
        const float* Xdata = X.data<float>();
        float* Ydata = Y->template mutable_data<float>();

        auto mask = Output(1, X.sizes(), at::dtype<bool>());
        bool* mask_data = mask->template mutable_data<bool>();
        auto* gen = context_.RandGenerator();
        for (int i = 0; i < X.numel(); ++i) {
          mask_data[i] = dist(gen) > 0.5;
          Ydata[i] = Xdata[i] * scale * mask_data[i];
        }
        return true;
      }
        */
    }
}

register_cpu_operator!{Dropout, DropoutOp<float, CPUContext>}
