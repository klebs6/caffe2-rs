crate::ix!();

impl ResizeNearest3DGradientOp<f32, CPUContext> {
    
    #[inline] pub fn run_on_device_with_ordernchw(&mut self) -> bool {
        
        todo!();
        /*
            const auto& dY = Input(0);
      const auto& X = Input(1);

      const auto inputDims = dY.sizes();
      CAFFE_ENFORCE_EQ(5, inputDims.size());
      const int batch_size = dY.dim32(0), num_channels = dY.dim32(1),
                input_frames = dY.dim32(2), input_height = dY.dim32(3),
                input_width = dY.dim32(4);

      const int output_frames = X.dim32(2);
      const int output_height = X.dim32(3);
      const int output_width = X.dim32(4);

      CAFFE_ENFORCE_EQ(InputSize(), 2);

      auto* dX = Output(
          0,
          {batch_size, num_channels, output_frames, output_height, output_width},
          at::dtype<float>());
      math::Set<float, CPUContext>(
          dX->numel(), 0.0f, dX->template mutable_data<float>(), &context_);

      const float* dYdata = dY.data<float>();
      float* dXdata = dX->template mutable_data<float>();

      for (int n = 0; n < batch_size; ++n) {
        for (int c = 0; c < num_channels; ++c) {
          for (int f = 0; f < input_frames; ++f) {
            const int out_f =
              std::min((int)(f / temporal_scale_), output_frames - 1);
            for (int y = 0; y < input_height; ++y) {
              const int out_y =
                  std::min((int)(y / height_scale_), (output_height - 1));
              for (int x = 0; x < input_width; ++x) {
                const int out_x =
                    std::min((int)(x / width_scale_), (output_width - 1));
                dXdata[(out_f * output_height + out_y) * output_width + out_x] +=
                  dYdata[(f * input_height + y) * input_width + x];
              }
            }
          }
          dYdata += input_frames * input_height * input_width;
          dXdata += output_frames * output_height * output_width;
        }
      }

      return true;
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            switch (order_) {
        case StorageOrder::NHWC:
          CAFFE_THROW("Not implemented for storage order: ", order_);
        case StorageOrder::NCHW:
          return RunOnDeviceWithOrderNCHW();
        default:
          CAFFE_THROW("Unknown Storage order: ", order_);
      }
        */
    }
}

register_cpu_gradient_operator!{
    ResizeNearest3DGradient,
    ResizeNearest3DGradientOp<float, CPUContext>
}
