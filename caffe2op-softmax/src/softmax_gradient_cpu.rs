crate::ix!();

/// Implementation for the CPU context.
impl SoftmaxGradientOp<f32, CPUContext> {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& Y = Input(0);
      auto& dY = Input(1);

      const auto canonical_axis = Y.canonical_axis_index(axis_);
      const int64_t N = Y.size_to_dim(canonical_axis);
      const int64_t D = Y.size_from_dim(canonical_axis);
      // First, get scales
      if (!scale_.defined()) {
        scale_ = caffe2::empty({N}, at::dtype<float>().device(CPU));
      } else if (scale_.numel() != N) {
        scale_.Resize(N);
      }

      if (!sum_multiplier_.defined()) {
        sum_multiplier_ = caffe2::empty({D}, at::dtype<float>().device(CPU));
        math::Set<float, CPUContext>(
            D, 1.f, sum_multiplier_.mutable_data<float>(), &context_);
      } else if (sum_multiplier_.numel() != D) {
        sum_multiplier_.Resize(D);
        math::Set<float, CPUContext>(
            D, 1.f, sum_multiplier_.mutable_data<float>(), &context_);
      }

      auto* dX = Output(0, Y.sizes(), at::dtype<float>());
      const float* Ydata = Y.data<float>();
      const float* dYdata = dY.data<float>();
      float* dXdata = dX->mutable_data<float>();
      if (N == 0 || D == 0) {
        return true;
      }
      context_.CopySameDevice<float>(Y.numel(), dYdata, dXdata);
      float* scaledata = scale_.mutable_data<float>();
      for (int i = 0; i < N; ++i) {
        math::Dot<float, CPUContext>(
            D, Ydata + i * D, dYdata + i * D, scaledata + i, &context_);
      }
      math::Gemm<float, CPUContext>(
          CblasNoTrans,
          CblasNoTrans,
          N,
          D,
          1,
          -1,
          scaledata,
          sum_multiplier_.data<float>(),
          1,
          dXdata,
          &context_);
      math::Mul<float, CPUContext>(Y.numel(), dXdata, Ydata, dXdata, &context_);
      return true;
        */
    }
}
