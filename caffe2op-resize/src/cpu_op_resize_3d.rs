crate::ix!();

impl ResizeNearest3DOp<f32, CPUContext> {

    #[inline] pub fn run_on_device_with_ordernchw(&mut self) -> bool {
        
        todo!();
        /*
            const auto& X = Input(0);
      const auto XDims = X.sizes();
      CAFFE_ENFORCE_EQ(5, XDims.size());

      const int batch_size = X.dim32(0), num_channels = X.dim32(1),
                input_frames = X.dim32(2), input_height = X.dim32(3),
                input_width = X.dim32(4);

      CAFFE_ENFORCE_EQ(InputSize(), 1);

      int output_frames = input_frames * temporal_scale_;
      int output_height = input_height * height_scale_;
      int output_width = input_width * width_scale_;
      auto* Y = Output(
          0,
          {batch_size, num_channels, output_frames, output_height, output_width},
          at::dtype<float>());

      const float* Xdata = X.data<float>();
      float* Ydata = Y->template mutable_data<float>();

      // Specialized implementation for fast 2x upsampling
      if (width_scale_ == 2.0 && height_scale_ == 2.0) {
        CAFFE_ENFORCE(temporal_scale_ == 1 || temporal_scale_ == 2,
          "temporal_scale must be either 1 or 2");

        resizeNearest3DNCHW2x(
            batch_size, num_channels, temporal_scale_, input_frames, input_height,
            input_width, Xdata, Ydata);
        return true;
      }

      CAFFE_THROW("Not implemented when height- and width scale are not 2");
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

register_cpu_operator!{ResizeNearest3D, ResizeNearest3DOp<float, CPUContext>}

#[cfg(caffe2_use_mkldnn)]
register_ideep_operator!{
    ResizeNearest3D,
    IDEEPFallbackOp<ResizeNearest3DOp<float, CPUContext>>
}

pub type ResizeNearest3DOpFloatCPU = ResizeNearest3DOp<f32,CPUContext>;

export_caffe2_op_to_c10_cpu!{
    ResizeNearest3D,
    "_caffe2::ResizeNearest3D(
        Tensor X, 
        str order, 
        float temporal_scale, 
        float width_scale, 
        float height_scale) -> (Tensor Y)",
        ResizeNearest3DOpFloatCPU
}
