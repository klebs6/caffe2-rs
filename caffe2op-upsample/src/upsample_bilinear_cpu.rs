crate::ix!();

impl UpsampleBilinearOp<f32, CPUContext> {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            const auto& X = Input(0);

      if (InputSize() == 2) {
        const auto& scales = Input(1);
        CAFFE_ENFORCE_EQ(scales.dim(), 1);
        CAFFE_ENFORCE_EQ(scales.numel(), 2);
        const float* scales_data = scales.data<float>();
        height_scale_ = scales_data[0];
        width_scale_ = scales_data[1];
      }

      const int batch_size = X.dim32(0);
      const int num_channels = X.dim32(1);
      const int input_height = X.dim32(2);
      const int input_width = X.dim32(3);
      int output_width = input_width * width_scale_;
      int output_height = input_height * height_scale_;
      auto* Y = Output(
          0,
          {batch_size, num_channels, output_height, output_width},
          at::dtype<float>());

      const float* input = X.data<float>();
      float* output = Y->mutable_data<float>();
      int channels = num_channels * batch_size;

      const float rheight = (output_height > 1)
          ? (float)(input_height - 1) / (output_height - 1)
          : 0.f;
      const float rwidth =
          (output_width > 1) ? (float)(input_width - 1) / (output_width - 1) : 0.f;
      for (int h2 = 0; h2 < output_height; ++h2) {
        const float h1r = rheight * h2;
        const int h1 = h1r;
        const int h1p = (h1 < input_height - 1) ? 1 : 0;
        const float h1lambda = h1r - h1;
        const float h0lambda = (float)1. - h1lambda;
        for (int w2 = 0; w2 < output_width; ++w2) {
          const float w1r = rwidth * w2;
          const int w1 = w1r;
          const int w1p = (w1 < input_width - 1) ? 1 : 0;
          const float w1lambda = w1r - w1;
          const float w0lambda = (float)1. - w1lambda;
          const float* Xdata = &input[h1 * input_width + w1];
          float* Ydata = &output[h2 * output_width + w2];
          for (int c = 0; c < channels; ++c) {
            Ydata[0] = h0lambda * (w0lambda * Xdata[0] + w1lambda * Xdata[w1p]) +
                h1lambda *
                    (w0lambda * Xdata[h1p * input_width] +
                     w1lambda * Xdata[h1p * input_width + w1p]);
            Xdata += input_width * input_height;
            Ydata += output_width * output_height;
          }
        }
      }

      return true;
        */
    }
}
