crate::ix!();

register_cpu_gradient_operator!{
    PadImageGradient, 
    PadImageGradientOp<f32, CPUContext>
}

impl PadImageGradientOp<f32, CPUContext> {

    #[inline] pub fn run_on_device_with_orderNCHW(&mut self) -> bool {
        
        todo!();
        /*
            auto& dY = Input(0);

      auto* dX = Output(
          0,
          {dY.dim32(0),
           dY.dim32(1),
           dY.dim32(2) - pad_t() - pad_b(),
           dY.dim32(3) - pad_l() - pad_r()},
          at::dtype<float>());
      int padded_height = dY.dim32(2);
      int padded_width = dY.dim32(3);
      int channels = dX->dim32(1);
      int height = dX->dim32(2);
      int width = dX->dim32(3);

      const float* dYdata = dY.data<float>();
      float* dXdata = dX->template mutable_data<float>();
      math::Set<float, CPUContext>(dX->numel(), 0, dXdata, &context_);
      // The main loop
      switch (mode_) {
        case PadMode::CONSTANT:
          for (int n = 0; n < dY.dim32(0); ++n) {
            for (int c = 0; c < channels; ++c) {
              for (int ph = 0; ph < padded_height; ++ph) {
                for (int pw = 0; pw < padded_width; ++pw) {
                  int h = ph - pad_t();
                  int w = pw - pad_l();
                  if (!(h < 0 || w < 0 || h >= height || w >= width)) {
                    dXdata[h * width + w] += dYdata[ph * padded_width + pw];
                  }
                }
              }
              // Do offset.
              dXdata += height * width;
              dYdata += padded_height * padded_width;
            }
          }
          break;
        case PadMode::REFLECT:
          for (int n = 0; n < dY.dim32(0); ++n) {
            for (int c = 0; c < channels; ++c) {
              for (int ph = 0; ph < padded_height; ++ph) {
                for (int pw = 0; pw < padded_width; ++pw) {
                  int h = ph - pad_t();
                  int w = pw - pad_l();
                  // max(h, -h) does reflection over 0
                  h = max(h, -h);
                  // min(h, 2 * height - h - 2) does reflection over height.
                  h = min(h, 2 * height - h - 2);
                  w = max(w, -w);
                  w = min(w, 2 * width - w - 2);
                  dXdata[h * width + w] += dYdata[ph * padded_width + pw];
                }
              }
              // Do offset.
              dXdata += height * width;
              dYdata += padded_height * padded_width;
            }
          }
          break;
        case PadMode::EDGE:
          for (int n = 0; n < dY.dim32(0); ++n) {
            for (int c = 0; c < channels; ++c) {
              for (int ph = 0; ph < padded_height; ++ph) {
                for (int pw = 0; pw < padded_width; ++pw) {
                  int h = min(height - 1, max(ph - pad_t(), 0));
                  int w = min(width - 1, max(pw - pad_l(), 0));
                  dXdata[h * width + w] += dYdata[ph * padded_width + pw];
                }
              }
              // Do offset.
              dXdata += height * width;
              dYdata += padded_height * padded_width;
            }
          }
          break;
      }
      return true;
        */
    }
    
    #[inline] pub fn run_on_device_with_orderNHWC(&mut self) -> bool {
        
        todo!();
        /*
            auto& dY = Input(0);

      auto* dX = Output(
          0,
          {dY.dim32(0),
           dY.dim32(1) - pad_t() - pad_b(),
           dY.dim32(2) - pad_l() - pad_r(),
           dY.dim32(3)},
          at::dtype<float>());
      int padded_height = dY.dim32(1);
      int padded_width = dY.dim32(2);
      int channels = dY.dim32(3);
      int height = dX->dim32(1);
      int width = dX->dim32(2);

      const float* dYdata = dY.data<float>();
      float* dXdata = dX->template mutable_data<float>();
      math::Set<float, CPUContext>(dX->numel(), 0, dXdata, &context_);

      switch (mode_) {
        case PadMode::CONSTANT:
          for (int n = 0; n < dY.dim32(0); ++n) {
            for (int ph = 0; ph < padded_height; ++ph) {
              for (int pw = 0; pw < padded_width; ++pw) {
                int h = ph - pad_t();
                int w = pw - pad_l();
                const int pad_index = (ph * padded_width + pw) * channels;
                if (!(h < 0 || w < 0 || h >= height || w >= width)) {
                  const int input_index = (h * width + w) * channels;
                  for (int c = 0; c < channels; ++c) {
                    dXdata[input_index + c] += dYdata[pad_index + c];
                  }
                }
              }
            }
            // Do offset.
            dXdata += dX->numel() / dX->dim32(0);
            dYdata += dY.numel() / dY.dim32(0);
          }
          break;
        case PadMode::REFLECT:
          for (int n = 0; n < dY.dim32(0); ++n) {
            for (int ph = 0; ph < padded_height; ++ph) {
              for (int pw = 0; pw < padded_width; ++pw) {
                const int pad_index = (ph * padded_width + pw) * channels;
                int h = ph - pad_t();
                int w = pw - pad_l();
                // max(h, -h) does reflection over 0
                h = max(h, -h);
                // min(h, 2 * height - h - 2) does reflection over height.
                h = min(h, 2 * height - h - 2);
                w = max(w, -w);
                w = min(w, 2 * width - w - 2);
                const int input_index = (h * width + w) * channels;
                for (int c = 0; c < channels; ++c) {
                  dXdata[input_index + c] += dYdata[pad_index + c];
                }
              }
            }
            // Do offset.
            dXdata += dX->numel() / dX->dim32(0);
            dYdata += dY.numel() / dY.dim32(0);
          }
          break;
        case PadMode::EDGE:
          for (int n = 0; n < dY.dim32(0); ++n) {
            for (int ph = 0; ph < padded_height; ++ph) {
              for (int pw = 0; pw < padded_width; ++pw) {
                const int pad_index = (ph * padded_width + pw) * channels;
                // Bounds to the right range.
                int h = min(height - 1, max(ph - pad_t(), 0));
                int w = min(width - 1, max(pw - pad_l(), 0));
                const int input_index = (h * width + w) * channels;
                for (int c = 0; c < channels; ++c) {
                  dXdata[input_index + c] += dYdata[pad_index + c];
                }
              }
            }
            // Do offset.
            dXdata += dX->numel() / dX->dim32(0);
            dYdata += dY.numel() / dY.dim32(0);
          }
          break;
      }
      return true;
        */
    }
}
