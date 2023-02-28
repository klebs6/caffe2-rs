crate::ix!();

impl ResizeNearestGradientOp<f32, CPUContext> {

    #[inline] pub fn run_on_device_with_order_nchw(&mut self) -> bool {
        
        todo!();
        /*
            const auto& dY = Input(0);
      const auto& X = Input(1);

      const auto inputDims = dY.sizes();
      CAFFE_ENFORCE_EQ(4, inputDims.size());
      const int batch_size = dY.dim32(0), num_channels = dY.dim32(1),
                input_height = dY.dim32(2), input_width = dY.dim32(3);
      const int output_height = X.dim32(2);
      const int output_width = X.dim32(3);
      if (InputSize() == 3) {
        const auto& scales = Input(2);
        CAFFE_ENFORCE_EQ(scales.dim(), 1);
        CAFFE_ENFORCE_EQ(scales.numel(), 2);
        const float* scales_data = scales.data<float>();
        height_scale_ = scales_data[0];
        width_scale_ = scales_data[1];
      }
      auto* dX = Output(
          0,
          {batch_size, num_channels, output_height, output_width},
          at::dtype<float>());
      math::Set<float, CPUContext>(
          dX->numel(), 0.0f, dX->template mutable_data<float>(), &context_);

      const float* dYdata = dY.data<float>();
      float* dXdata = dX->template mutable_data<float>();

      for (int n = 0; n < batch_size; ++n) {
        for (int c = 0; c < num_channels; ++c) {
          for (int y = 0; y < input_height; ++y) {
            const int out_y =
                std::min((int)(y / height_scale_), (output_height - 1));
            for (int x = 0; x < input_width; ++x) {
              const int out_x =
                  std::min((int)(x / width_scale_), (output_width - 1));
              dXdata[output_width * out_y + out_x] += dYdata[input_width * y + x];
            }
          }
          dYdata += input_height * input_width;
          dXdata += output_height * output_width;
        }
      }

      return true;
        */
    }
    
    #[inline] pub fn run_on_device_with_order_nhwc(&mut self) -> bool {
        
        todo!();
        /*
            const auto& dY = Input(0);
      const auto& X = Input(1);

      const auto inputDims = dY.sizes();
      CAFFE_ENFORCE_EQ(4, inputDims.size());
      const int batch_size = dY.dim32(0), input_height = dY.dim32(1),
                input_width = dY.dim32(2), num_channels = dY.dim32(3);
      const int output_height = X.dim32(1);
      const int output_width = X.dim32(2);
      if (InputSize() == 3) {
        const auto& scales = Input(2);
        CAFFE_ENFORCE_EQ(scales.dim(), 1);
        CAFFE_ENFORCE_EQ(scales.numel(), 2);
        const float* scales_data = scales.data<float>();
        height_scale_ = scales_data[0];
        width_scale_ = scales_data[1];
      }
      auto* dX = Output(
          0,
          {batch_size, output_height, output_width, num_channels},
          at::dtype<float>());
      math::Set<float, CPUContext>(
          dX->numel(), 0.0f, dX->template mutable_data<float>(), &context_);

      const int output_width_stride = output_width * num_channels;
      const int input_width_stride = input_width * num_channels;

      const float* dYdata = dY.data<float>();
      float* dXdata = dX->template mutable_data<float>();

      for (int n = 0; n < batch_size; ++n) {
        for (int y = 0; y < input_height; ++y) {
          const int out_y = std::min((int)(y / height_scale_), (output_height - 1));
          for (int x = 0; x < input_width; ++x) {
            const int out_x = std::min((int)(x / width_scale_), (output_width - 1));

            float* dXdata_c0 =
                dXdata + output_width_stride * out_y + num_channels * out_x;
            const float* dYdata_c0 =
                dYdata + input_width_stride * y + num_channels * x;

            for (int c = 0; c < num_channels; ++c) {
              dXdata_c0[c] += dYdata_c0[c];
            }
          }
        }
        dYdata += input_height * input_width_stride;
        dXdata += output_height * output_width_stride;
      }

      return true;
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            switch (order_) {
        case StorageOrder::NHWC:
          return RunOnDeviceWithOrderNHWC();
        case StorageOrder::NCHW:
          return RunOnDeviceWithOrderNCHW();
        default:
          CAFFE_THROW("Unknown Storage order: ", order_);
      }
        */
    }
}

register_cpu_gradient_operator!{
    ResizeNearestGradient,
    ResizeNearestGradientOp<f32, CPUContext>
}
