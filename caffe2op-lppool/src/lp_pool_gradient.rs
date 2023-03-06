crate::ix!();

///-------------------------------------
register_cpu_operator!{
    LpPoolGradient, 
    PoolGradientOp<float, CPUContext, LpPoolFunctor>
}

num_inputs!{LpPoolGradient, 3}

num_outputs!{LpPoolGradient, 1}

pub struct LpPoolGradientOp(PoolGradientOp<f32, CPUContext, LpPoolFunctor>);

impl LpPoolGradientOp {
    
    #[inline] pub fn run_f32_on_cpu_device_with_order_nchw(&mut self) -> bool {
        
        todo!();
        /*
            const auto& X = Input(0);
      const auto& Y = Input(1);
      auto& dY = Input(2);

      const auto p = OperatorStorage::GetSingleArgument<float>("p", 2.0);

      // TODO(Yangqing): Add shape checks.
      auto* dX = Output(0, X.sizes(), at::dtype<float>());
      math::Set<float, CPUContext>(
          X.numel(), 0, dX->template mutable_data<float>(), &context_);
      const float* dYdata = dY.data<float>();
      const float* Xdata = X.data<float>();
      const float* Ydata = Y.data<float>();
      float* dXdata = dX->template mutable_data<float>();

      int channels = X.dim32(1);
      CAFFE_ENFORCE_EQ(channels, dY.dim32(1));
      int height = X.dim32(2);
      int width = X.dim32(3);
      ConvPoolOpBase<CPUContext>::ComputePads({height, width});
      int pooled_height = dY.dim32(2);
      int pooled_width = dY.dim32(3);
      // The main loop
      for (int n = 0; n < X.dim32(0); ++n) {
        for (int c = 0; c < channels; ++c) {
          for (int ph = 0; ph < pooled_height; ++ph) {
            for (int pw = 0; pw < pooled_width; ++pw) {
              int hstart = ph * stride_[0] - pads_[0];
              int wstart = pw * stride_[1] - pads_[1];
              int hend = min(hstart + kernel_[0], height);
              int wend = min(wstart + kernel_[1], width);
              hstart = max(hstart, 0);
              wstart = max(wstart, 0);
              for (int h = hstart; h < hend; ++h) {
                for (int w = wstart; w < wend; ++w) {
                  // gradient of p-norm is x_j * |x_j|^{p-2} / |x|_p^{p-1}
                  dXdata[h * width + w] += dYdata[ph * pooled_width + pw] *
                      Xdata[h * width + w] *
                      std::pow(std::abs(Xdata[h * width + w]), p - 2) /
                      std::pow(Ydata[ph * pooled_width + pw], p - 1);
                }
              }
            }
          }
          // offset
          dXdata += height * width;
          dYdata += pooled_height * pooled_width;
          Ydata += pooled_height * pooled_width;
          Xdata += height * width;
        }
      }
      return true;
        */
    }
    
    #[inline] pub fn run_f32_on_cpu_device_with_order_nhwc(&mut self) -> bool {
        
        todo!();
        /*
            const auto& X = Input(0);
      const auto& Y = Input(1);
      auto& dY = Input(2);
      CAFFE_ENFORCE_EQ(dY.dim(), 4);

      // TODO(Yangqing): Add shape checks.
      auto* dX = Output(0, X.sizes(), at::dtype<float>());
      math::Set<float, CPUContext>(
          X.numel(), 0, dX->template mutable_data<float>(), &context_);
      const float* dYdata = dY.data<float>();
      float* dXdata = dX->template mutable_data<float>();
      const float* Xdata = X.data<float>();
      const float* Ydata = Y.data<float>();
      // The main loop
      int height = X.dim32(1);
      int width = X.dim32(2);
      ConvPoolOpBase<CPUContext>::ComputePads({height, width});
      const auto p = OperatorStorage::GetSingleArgument<float>("p", 2.0);

      int pooled_height = dY.dim32(1);
      int pooled_width = dY.dim32(2);
      int channels = X.dim32(3);
      CAFFE_ENFORCE_EQ(channels, dY.dim32(3));
      for (int n = 0; n < X.dim32(0); ++n) {
        for (int ph = 0; ph < pooled_height; ++ph) {
          for (int pw = 0; pw < pooled_width; ++pw) {
            int hstart = ph * stride_[0] - pads_[0];
            int wstart = pw * stride_[1] - pads_[1];
            int hend = min(hstart + kernel_[0], height);
            int wend = min(wstart + kernel_[1], width);
            hstart = max(hstart, 0);
            wstart = max(wstart, 0);
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                for (int c = 0; c < channels; ++c) {
                  dXdata[(h * width + w) * channels + c] +=
                      dYdata[(ph * pooled_width + pw) * channels + c] *
                      Xdata[(h * width + w) * channels + c] *
                      std::pow(
                          std::abs(Xdata[(h * width + w) * channels + c]), p - 2) /
                      std::pow(
                          Ydata[(ph * pooled_width + pw) * channels + c], p - 1);
                }
              }
            }
          }
        }
        // offset
        dXdata += X.numel() / X.dim32(0);
        dYdata += dY.numel() / dY.dim32(0);
        Xdata += X.numel() / X.dim32(0);
        Ydata += Y.numel() / Y.dim32(0);
      }
      return true;
        */
    }
}
