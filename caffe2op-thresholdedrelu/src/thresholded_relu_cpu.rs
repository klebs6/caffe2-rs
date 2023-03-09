crate::ix!();

impl ThresholdedReluOp<f32,CPUContext> {

    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& X = Input(0);

      auto* Y = Output(0, X.sizes(), at::dtype<float>());

      ConstEigenVectorArrayMap<float> Xvec(X.data<float>(), X.numel());
      EigenVectorArrayMap<float> Yvec(
          Y->template mutable_data<float>(), Y->numel());
      Yvec = (Xvec > alpha_).select(Xvec, 0.f);
      /* Naive implementation
      const float* Xdata = X.data<float>();
      float* Ydata = Y->template mutable_data<float>();
      for (int i = 0; i < X.size(); ++i) {
        Xdata[i] -= alpha_;
        Ydata[i] = std::max(Xdata[i], 0.0f);
      }
      */
      return true;
        */
    }
}
