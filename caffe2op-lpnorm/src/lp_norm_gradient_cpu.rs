crate::ix!();

impl LpNormGradientOp<f32, CPUContext> {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            const auto& X = Input(0);
      const auto& dnorm = Input(1);

      CAFFE_ENFORCE_EQ(dnorm.dim(), 1);
      CAFFE_ENFORCE_EQ(dnorm.dim32(0), 1);
      auto* dX = Output(0, X.sizes(), at::dtype<float>());
      const float size = average_ ? (float)X.numel() : 1.0f;
      if (p_ == 1) {
        EigenVectorMap<float>(dX->template mutable_data<float>(), X.numel())
            .array() = ConstEigenVectorMap<float>(X.data<float>(), X.numel())
                           .array()
                           .unaryExpr([](float x) {
                             const float kEps = 1e-12f;
                             if (x < -kEps) {
                               return -1.0f;
                             } else if (x > kEps) {
                               return 1.0f;
                             } else {
                               return 0.0f;
                             }
                           }) *
            ((dnorm.data<float>())[0] / size);
      } else if (p_ == 2) {
        EigenVectorMap<float>(dX->template mutable_data<float>(), X.numel())
            .array() =
            ConstEigenVectorMap<float>(X.data<float>(), X.numel()).array() * 2.0f *
            ((dnorm.data<float>())[0] / size);
      }

      return true;
        */
    }
}

// LpNorm
register_cpu_operator!{LpNorm,         LpNormOp<f32, CPUContext>}
register_cpu_operator!{LpNormGradient, LpNormGradientOp<f32, CPUContext>}
