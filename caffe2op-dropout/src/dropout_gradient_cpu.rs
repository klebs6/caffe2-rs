crate::ix!();

impl DropoutGradientOp<f32, CPUContext> {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& dY = Input(0);

      auto* dX = Output(0, dY.sizes(), at::dtype<float>());
      if (is_test_) {
        if (dX != &dY) {
          context_.CopyFromCPU<float>(
              dY.numel(), dY.data<float>(), dX->template mutable_data<float>());
        }
        return true;
      } else {
        auto& mask = Input(1);
        CAFFE_ENFORCE_EQ(dY.numel(), mask.numel());
        const float* dYdata = dY.data<float>();
        const bool* mask_data = mask.data<bool>();
        float* dXdata = dX->template mutable_data<float>();
        float scale = 1. / (1. - ratio_);
        for (int i = 0; i < dY.numel(); ++i) {
          dXdata[i] = dYdata[i] * mask_data[i] * scale;
        }
        return true;
      }
        */
    }
}

