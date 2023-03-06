crate::ix!();

pub struct LpPoolOp(PoolOp<f32, CPUContext, LpPoolFunctor>);

impl LpPoolOp {

    #[inline] pub fn run_f32_on_cpu_device_with_order_nchw(&mut self) -> bool {
        
        todo!();
        /*
            auto& X = Input(0);
      auto* Y = Output(0);
      ConvPoolOpBase::SetOutputSize(X, Y, X.dim32(1));
      const auto p = OperatorStorage::GetSingleArgument<float>("p", 2.0);
      const auto inv_p = 1.0 / p;

      const float* Xdata = X.data<float>();
      float* Ydata = Y->template mutable_data<float>();
      math::Set<float, CPUContext>(Y->numel(), 0, Ydata, &context_);
      // The main loop
      int channels = X.dim32(1);
      int height = X.dim32(2);
      int width = X.dim32(3);
      int pooled_height = Y->dim32(2);
      int pooled_width = Y->dim32(3);

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
              const int pool_index = ph * pooled_width + pw;
              for (int h = hstart; h < hend; ++h) {
                for (int w = wstart; w < wend; ++w) {
                  const int input_index = h * width + w;
                  Ydata[pool_index] += std::pow(std::abs(Xdata[input_index]), p);
                }
              }
              Ydata[pool_index] = std::pow(Ydata[pool_index], inv_p);
            }
          }
          // Do offset.
          Xdata += height * width;
          Ydata += pooled_height * pooled_width;
        }
      }
      return true;
        */
    }
    
    #[inline] pub fn run_f32_on_cpu_device_with_order_nhwc(&mut self) -> bool {
        
        todo!();
        /*
            auto& X = Input(0);
      auto* Y = Output(0);
      int height = X.dim32(1);
      int width = X.dim32(2);
      int channels = X.dim32(3);
      ConvPoolOpBase::SetOutputSize(X, Y, channels);

      const auto p = OperatorStorage::GetSingleArgument<float>("p", 2.0);
      const auto inv_p = 1.0 / p;

      const float* Xdata = X.data<float>();
      float* Ydata = Y->template mutable_data<float>();
      math::Set<float, CPUContext>(Y->numel(), 0, Ydata, &context_);
      // The main loop
      int pooled_height = Y->dim32(1);
      int pooled_width = Y->dim32(2);
      for (int n = 0; n < X.dim32(0); ++n) {
        for (int ph = 0; ph < pooled_height; ++ph) {
          for (int pw = 0; pw < pooled_width; ++pw) {
            int hstart = ph * stride_[0] - pads_[0];
            int wstart = pw * stride_[1] - pads_[1];
            int hend = min(hstart + kernel_[0], height);
            int wend = min(wstart + kernel_[1], width);
            hstart = max(hstart, 0);
            wstart = max(wstart, 0);
            const int pool_index = (ph * pooled_width + pw) * channels;
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                const int input_index = (h * width + w) * channels;
                for (int c = 0; c < channels; ++c) {
                  Ydata[pool_index + c] +=
                      std::pow(std::abs(Xdata[input_index + c]), p);
                }
              }
            }
            for (int c = 0; c < channels; ++c) {
              Ydata[pool_index + c] = std::pow(Ydata[pool_index + c], inv_p);
            }
          }
        }
        // Do offset.
        Xdata += X.numel() / X.dim32(0);
        Ydata += Y->numel() / Y->dim32(0);
      }
      return true;
        */
    }
}
