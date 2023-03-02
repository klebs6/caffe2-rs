crate::ix!();

impl DotProductGradientOp<f32, CPUContext> {

    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& X = Input(X_IN);
      auto& Y = Input(Y_IN);
      auto& dDot = Input(DER_DOT_IN);

      int N, D;
      if (X.numel() > 0) {
        N = X.dim() > 0 ? X.dim32(0) : 1;
        D = X.numel() / N;
      } else {
        N = 0;
        D = 0;
      }
      CAFFE_ENFORCE(X.dim() == Y.dim());
      for (int i = 0; i < X.dim(); ++i) {
        CAFFE_ENFORCE(X.dim32(i) == Y.dim32(i));
      }
      CAFFE_ENFORCE(dDot.dim() == 1);
      CAFFE_ENFORCE(dDot.dim32(0) == N);
      auto* dX = Output(DER_X_OUT, X.sizes(), at::dtype<float>());
      auto* dY = Output(DER_Y_OUT, Y.sizes(), at::dtype<float>());

      const auto* X_data = X.template data<float>();
      const auto* Y_data = Y.template data<float>();
      const auto* dDot_data = dDot.template data<float>();
      auto* dX_data = dX->template mutable_data<float>();
      auto* dY_data = dY->template mutable_data<float>();
      for (int i = 0; i < N; ++i) { // TODO: multithreading
        auto offset = i * D;
        math::Scale<float, float, CPUContext>(
            D, dDot_data[i], X_data + offset, dY_data + offset, &context_);
        math::Scale<float, float, CPUContext>(
            D, dDot_data[i], Y_data + offset, dX_data + offset, &context_);
      }
      return true;
        */
    }
}
