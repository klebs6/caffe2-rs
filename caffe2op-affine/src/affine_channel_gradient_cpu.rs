crate::ix!();

impl AffineChannelGradientOp<f32, CPUContext> {

    #[inline] pub fn run_on_device_with_order_nchw(&mut self) -> bool {
        
        todo!();
        /*
            const auto& dY = Input(0);
      const auto& scale = is_learnable_ ? Input(2) : Input(1);

      auto* dX = Output(0, dY.sizes(), at::dtype<float>());
      const int N = dY.dim32(0);
      const int C = dY.dim32(1);
      const int HxW = dY.numel() / (N * C);
      const float* dY_data = dY.data<float>();
      const float* scale_data = scale.data<float>();
      const std::array<int, 3> X_dims = {N, C, HxW};
      const std::array<int, 3> scale_dims = {1, C, 1};
      math::Mul<float, CPUContext>(
          3,
          X_dims.data(),
          3,
          scale_dims.data(),
          dY_data,
          scale_data,
          dX->template mutable_data<float>(),
          &context_);
      if (is_learnable_) {
        const auto& X = Input(1);
        const float* X_data = X.data<float>();

        auto* dscale = Output(1, scale.sizes(), at::dtype<float>());
        auto* dbias = Output(2, scale.sizes(), at::dtype<float>());
        AffineChannelScaleBiasBackwardNCHW<float>(
            N,
            C,
            HxW,
            dY_data,
            X_data,
            dscale->template mutable_data<float>(),
            dbias->template mutable_data<float>());
      }
      return true;
        */
    }
    
    #[inline] pub fn run_on_device_with_order_nhwc(&mut self) -> bool {
        
        todo!();
        /*
            const auto& dY = Input(0);
      const auto& scale = is_learnable_ ? Input(2) : Input(1);

      auto* dX = Output(0, dY.sizes(), at::dtype<float>());
      const int ndim = dY.dim();
      const int C = dY.dim32(ndim - 1);
      const int rows = dY.numel() / C;
      const int cols = C;
      const float* dY_data = dY.data<float>();
      const float* scale_data = scale.data<float>();
      math::RowwiseMul<float, CPUContext>(
          rows,
          cols,
          dY_data,
          scale_data,
          dX->template mutable_data<float>(),
          &context_);
      if (is_learnable_) {
        const auto& X = Input(1);
        const float* X_data = X.data<float>();
        const int N = X.dim32(0);
        const int HxW = rows / N;

        auto* dscale = Output(1, scale.sizes(), at::dtype<float>());
        auto* dbias = Output(2, scale.sizes(), at::dtype<float>());
        AffineChannelScaleBiasBackwardNHWC<float>(
            N,
            C,
            HxW,
            dY_data,
            X_data,
            dscale->template mutable_data<float>(),
            dbias->template mutable_data<float>());
      }
      return true;
        */
    }
}
