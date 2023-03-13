crate::ix!();

impl LeakyReluGradientOp<f32, CPUContext> {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            const auto& Y = Input(0);
      const auto& dY = Input(1);

      auto* dX = Output(0, Y.sizes(), at::dtype<float>());
      CAFFE_ENFORCE_EQ(Y.numel(), dY.numel());
      ConstEigenVectorMap<float> Yvec(Y.template data<float>(), Y.numel());
      ConstEigenVectorMap<float> dYvec(dY.template data<float>(), dY.numel());
      EigenVectorMap<float> dXvec(dX->template mutable_data<float>(), dX->numel());
      Eigen::VectorXf gtZero = (Yvec.array() >= 0.0f).cast<float>();
      dXvec = dYvec.array() * gtZero.array() -
          dYvec.array() * (gtZero.array() - 1.0f) * alpha_;
      return true;
        */
    }
}
