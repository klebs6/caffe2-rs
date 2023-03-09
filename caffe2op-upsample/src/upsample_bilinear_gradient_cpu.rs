crate::ix!();

register_cpu_operator!{
    UpsampleBilinear,         
    UpsampleBilinearOp<f32, CPUContext>
}

register_cpu_operator!{
    UpsampleBilinearGradient, 
    UpsampleBilinearGradientOp<f32, CPUContext>
}

impl UpsampleBilinearGradientOp<f32, CPUContext> {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
      const auto& dY = Input(0);
      const auto& X  = Input(1);

      if (InputSize() == 3) {
        const auto& scales = Input(2);
        CAFFE_ENFORCE_EQ(scales.dim(), 1);
        CAFFE_ENFORCE_EQ(scales.numel(), 2);
        const float* scales_data = scales.data<float>();
        height_scale_ = scales_data[0];
        width_scale_ = scales_data[1];
      }

      const auto inputDims = dY.sizes();
      CAFFE_ENFORCE_EQ(4, inputDims.size());
      const int batch_size = dY.dim32(0);
      const int num_channels = dY.dim32(1);
      const int input_height = dY.dim32(2);
      const int input_width = dY.dim32(3);
      const int output_height = X.dim32(2);
      const int output_width = X.dim32(3);
      auto* dX = Output(
          0,
          {batch_size, num_channels, output_height, output_width},
          at::dtype<float>());
      math::Set<float, CPUContext>(
          dX->numel(), 0.0f, dX->mutable_data<float>(), &context_);

      const float* dYdata = dY.data<float>();
      float* dXdata = dX->mutable_data<float>();
      int channels = num_channels * batch_size;

      const float rheight = (input_height > 1)
          ? (float)(output_height - 1) / (input_height - 1)
          : 0.f;
      const float rwidth =
          (input_width > 1) ? (float)(output_width - 1) / (input_width - 1) : 0.f;

      for (int h2 = 0; h2 < input_height; ++h2) {
        const float h1r = rheight * h2;
        const int h1 = h1r;
        const int h1p = (h1 < output_height - 1) ? 1 : 0;
        const float h1lambda = h1r - h1;
        const float h0lambda = (float)1. - h1lambda;
        for (int w2 = 0; w2 < input_width; ++w2) {
          const float w1r = rwidth * w2;
          const int w1 = w1r;
          const int w1p = (w1 < output_width - 1) ? 1 : 0;
          const float w1lambda = w1r - w1;
          const float w0lambda = (float)1. - w1lambda;
          float* pos1 = &dXdata[h1 * output_width + w1];
          const float* pos2 = &dYdata[h2 * input_width + w2];
          for (int c = 0; c < channels; ++c) {
            pos1[0] += h0lambda * w0lambda * pos2[0];
            pos1[w1p] += h0lambda * w1lambda * pos2[0];
            pos1[h1p * output_width] += h1lambda * w0lambda * pos2[0];
            pos1[h1p * output_width + w1p] += h1lambda * w1lambda * pos2[0];
            pos1 += output_width * output_height;
            pos2 += input_width * input_height;
          }
        }
      }

      return true;
        */
    }
}
