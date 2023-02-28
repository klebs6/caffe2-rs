crate::ix!();

impl ResizeNearestOp<f32, CPUContext> {

    #[inline] pub fn run_on_device_with_ordernchw(&mut self) -> bool {
        
        todo!();
        /*
            const auto& X = Input(0);

      const int batch_size = X.dim32(0), num_channels = X.dim32(1),
                input_height = X.dim32(2), input_width = X.dim32(3);
      if (InputSize() == 2) {
        const auto& scales = Input(1);
        CAFFE_ENFORCE_EQ(scales.dim(), 1);
        CAFFE_ENFORCE_EQ(scales.numel(), 2);
        const float* scales_data = scales.data<float>();
        height_scale_ = scales_data[0];
        width_scale_ = scales_data[1];
      }

      int output_width = input_width * width_scale_;
      int output_height = input_height * height_scale_;
      auto* Y = Output(
          0,
          {batch_size, num_channels, output_height, output_width},
          at::dtype<float>());

      const float* Xdata = X.data<float>();
      float* Ydata = Y->template mutable_data<float>();

      // Specialized implementation for fast 2x upsampling
      if (width_scale_ == 2.0 && height_scale_ == 2.0) {
        resizeNearestNCHW2x(
            batch_size, num_channels, input_height, input_width, Xdata, Ydata);
        return true;
      }

      for (int n = 0; n < batch_size; ++n) {
        for (int c = 0; c < num_channels; ++c) {
          for (int y = 0; y < output_height; ++y) {
            const int in_y = std::min((int)(y / height_scale_), (input_height - 1));
            for (int x = 0; x < output_width; ++x) {
              const int in_x = std::min((int)(x / width_scale_), (input_width - 1));
              Ydata[output_width * y + x] = Xdata[input_width * in_y + in_x];
            }
          }
          Xdata += input_height * input_width;
          Ydata += output_width * output_height;
        }
      }

      return true;
        */
    }
    
    #[inline] pub fn run_on_device_with_ordernhwc(&mut self) -> bool {
        
        todo!();
        /*
            const auto& X = Input(0);

      const int batch_size = X.dim32(0), input_height = X.dim32(1),
                input_width = X.dim32(2), num_channels = X.dim32(3);
      if (InputSize() == 2) {
        const auto& scales = Input(1);
        CAFFE_ENFORCE_EQ(scales.dim(), 1);
        CAFFE_ENFORCE_EQ(scales.numel(), 2);
        const float* scales_data = scales.data<float>();
        height_scale_ = scales_data[0];
        width_scale_ = scales_data[1];
      }

      int output_width = input_width * width_scale_;
      int output_height = input_height * height_scale_;

      const int output_width_stride = output_width * num_channels;
      const int input_width_stride = input_width * num_channels;

      auto* Y = Output(
          0,
          {batch_size, output_height, output_width, num_channels},
          at::dtype<float>());

      const float* Xdata = X.data<float>();
      float* Ydata = Y->template mutable_data<float>();

      for (int n = 0; n < batch_size; ++n) {
        for (int y = 0; y < output_height; ++y) {
          const int in_y = std::min((int)(y / height_scale_), (input_height - 1));
          for (int x = 0; x < output_width; ++x) {
            const int in_x = std::min((int)(x / width_scale_), (input_width - 1));
            std::memcpy(
                &Ydata[output_width_stride * y + num_channels * x],
                &Xdata[input_width_stride * in_y + num_channels * in_x],
                num_channels * sizeof(float));
          }
        }
        Xdata += input_height * input_width_stride;
        Ydata += output_height * output_width_stride;
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

register_cpu_operator!{
    ResizeNearest, 
    ResizeNearestOp<f32, CPUContext>
}

#[cfg(feature = "mkldnn")]
register_ideep_operator!{
    ResizeNearest,
    IDEEPFallbackOp<ResizeNearestOp<f32, CPUContext>>
}

pub type ResizeNearestOpFloatCPU = ResizeNearestOp<f32,CPUContext>;

export_caffe2_op_to_c10_cpu!{ResizeNearest,
    "_caffe2::ResizeNearest(
        Tensor X, 
        str order, 
        float width_scale, 
        float height_scale) -> (Tensor Y)",
    ResizeNearestOpFloatCPU}
