crate::ix!();

register_cpu_operator!{
    PadImage, 
    PadImageOp<f32, CPUContext>
}

impl PadImageOp<f32, CPUContext> {

    #[inline] pub fn pad_tensor_inference(
        &mut self, 
        def:   &OperatorDef, 
        input: &Vec<TensorShape>) -> Vec<TensorShape> {
        
        todo!();
        /*
            return ConvPoolOpBase::TensorInferenceForPool(def, in);
        */
    }
    
    #[inline] pub fn run_on_device_with_orderNCHW(&mut self) -> bool {
        
        todo!();
        /*
            auto& X = Input(0);
      auto* Y = Output(0);
      int channels = X.dim32(1);
      int height = X.dim32(2);
      int width = X.dim32(3);
      ConvPoolOpBase::SetOutputSize(X, Y, channels);

      const float* Xdata = X.data<float>();
      float* Ydata = Y->template mutable_data<float>();
      // The main loop
      int padded_height = Y->dim32(2);
      int padded_width = Y->dim32(3);

      switch (mode_) {
        case PadMode::CONSTANT:
          for (int n = 0; n < X.dim32(0); ++n) {
            for (int c = 0; c < channels; ++c) {
              for (int ph = 0; ph < padded_height; ++ph) {
                for (int pw = 0; pw < padded_width; ++pw) {
                  int h = ph - pad_t();
                  int w = pw - pad_l();
                  Ydata[ph * padded_width + pw] =
                      (h < 0 || w < 0 || h >= height || w >= width)
                      ? value_
                      : Xdata[h * width + w];
                }
              }
              // Do offset.
              Xdata += height * width;
              Ydata += padded_height * padded_width;
            }
          }
          break;
        case PadMode::REFLECT:
          if (pad_r() >= 0 && pad_t() >= 0 && pad_l() >= 0 && pad_b() >= 0) {
            for (int n = 0; n < X.dim32(0); ++n) {
              for (int c = 0; c < channels; ++c) {
                // Handle the valid region:
                // i.e. Y[n][c][pad_t:pad_t+h][pad_l:pad_l+w]
                auto* Ystart = Ydata + pad_t() * padded_width + pad_l();
                math::CopyMatrix<CPUContext>(
                    sizeof(float),
                    height,
                    width,
                    Xdata,
                    width,
                    Ystart,
                    padded_width,
                    &context_);

    // Fixup areas where we need to reflect
    #define X(ph, pw)                 \
      int h = ph - pad_t();           \
      int w = pw - pad_l();           \
      h = max(h, -h);                 \
      h = min(h, 2 * height - h - 2); \
      w = max(w, -w);                 \
      w = min(w, 2 * width - w - 2);  \
      Ydata[ph * padded_width + pw] = Xdata[h * width + w]

                // Top part
                for (int ph = 0; ph < pad_t(); ++ph) {
                  for (int pw = 0; pw < padded_width; ++pw) {
                    X(ph, pw);
                  }
                }

                // Bottom part
                for (int ph = padded_height - pad_b(); ph < padded_height; ++ph) {
                  for (int pw = 0; pw < padded_width; ++pw) {
                    X(ph, pw);
                  }
                }

                // Interior
                for (int ph = pad_t(); ph < padded_height - pad_b(); ++ph) {
                  // Left
                  for (int pw = 0; pw < pad_l(); ++pw) {
                    X(ph, pw);
                  }
                  // Right
                  for (int pw = padded_width - pad_r(); pw < padded_width; ++pw) {
                    X(ph, pw);
                  }
                }
    #undef X

                // Do offset.
                Xdata += height * width;
                Ydata += padded_height * padded_width;
              }
            }
          } else {
            for (int n = 0; n < X.dim32(0); ++n) {
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
                    Ydata[ph * padded_width + pw] = Xdata[h * width + w];
                  }
                }
                // Do offset.
                Xdata += height * width;
                Ydata += padded_height * padded_width;
              }
            }
          }
          break;
        case PadMode::EDGE:
          for (int n = 0; n < X.dim32(0); ++n) {
            for (int c = 0; c < channels; ++c) {
              for (int ph = 0; ph < padded_height; ++ph) {
                for (int pw = 0; pw < padded_width; ++pw) {
                  // Bounds to the right range.
                  int h = min(height - 1, max(ph - pad_t(), 0));
                  int w = min(width - 1, max(pw - pad_l(), 0));
                  Ydata[ph * padded_width + pw] = Xdata[h * width + w];
                }
              }
              // Do offset.
              Xdata += height * width;
              Ydata += padded_height * padded_width;
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
            auto& X = Input(0);
      auto* Y = Output(0);
      int height = X.dim32(1);
      int width = X.dim32(2);
      int channels = X.dim32(3);
      ConvPoolOpBase::SetOutputSize(X, Y, channels);
      const float* Xdata = X.data<float>();
      float* Ydata = Y->template mutable_data<float>();

      // The main loop
      int padded_height = Y->dim32(1);
      int padded_width = Y->dim32(2);

      switch (mode_) {
        case PadMode::CONSTANT:
          for (int n = 0; n < X.dim32(0); ++n) {
            for (int ph = 0; ph < padded_height; ++ph) {
              for (int pw = 0; pw < padded_width; ++pw) {
                int h = ph - pad_t();
                int w = pw - pad_l();
                const int pad_index = (ph * padded_width + pw) * channels;
                if (h < 0 || w < 0 || h >= height || w >= width) {
                  for (int c = 0; c < channels; ++c) {
                    Ydata[pad_index + c] = value_;
                  }
                } else {
                  const int input_index = (h * width + w) * channels;
                  for (int c = 0; c < channels; ++c) {
                    Ydata[pad_index + c] = Xdata[input_index + c];
                  }
                }
              }
            }
            // Do offset.
            Xdata += X.numel() / X.dim32(0);
            Ydata += Y->numel() / Y->dim32(0);
          }
          break;
        case PadMode::REFLECT:
          for (int n = 0; n < X.dim32(0); ++n) {
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
                  Ydata[pad_index + c] = Xdata[input_index + c];
                }
              }
            }
            // Do offset.
            Xdata += X.numel() / X.dim32(0);
            Ydata += Y->numel() / Y->dim32(0);
          }
          break;
        case PadMode::EDGE:
          for (int n = 0; n < X.dim32(0); ++n) {
            for (int ph = 0; ph < padded_height; ++ph) {
              for (int pw = 0; pw < padded_width; ++pw) {
                const int pad_index = (ph * padded_width + pw) * channels;
                int h = min(height - 1, max(ph - pad_t(), 0));
                int w = min(width - 1, max(pw - pad_l(), 0));
                const int input_index = (h * width + w) * channels;
                for (int c = 0; c < channels; ++c) {
                  Ydata[pad_index + c] = Xdata[input_index + c];
                }
              }
            }
            // Do offset.
            Xdata += X.numel() / X.dim32(0);
            Ydata += Y->numel() / Y->dim32(0);
          }
          break;
      }
      return true;
        */
    }
}
