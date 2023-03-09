crate::ix!();

register_cpu_operator!{Selu, SeluOp<float, CPUContext>}

register_cpu_operator!{SeluGradient, SeluGradientOp<float, CPUContext>}

impl SeluOp<f32, CPUContext> {

    #[inline] pub fn run_on_deviceA(&mut self) -> bool {
        
        todo!();
        /*
            auto& X = Input(0);

      auto* Y = Output(0, X.sizes(), at::dtype<float>());

      ConstEigenVectorArrayMap<float> Xvec(X.data<float>(), X.numel());
      EigenVectorArrayMap<float> Yvec(
          Y->template mutable_data<float>(), Y->numel());
      Yvec = lambda_ * (Xvec > 0).select(Xvec, (alpha_ * Xvec.exp() - alpha_));
      return true;
        */
    }
    
    #[inline] pub fn run_on_deviceB(&mut self) -> bool {
        
        todo!();
        /*
            auto& Y = Input(0);
      auto& dY = Input(1);

      CAFFE_ENFORCE_EQ(dY.numel(), Y.numel());
      auto* dX = Output(0, Y.sizes(), at::dtype<float>());

      ConstEigenVectorArrayMap<float> Yvec(Y.data<float>(), Y.numel());
      ConstEigenVectorArrayMap<float> dYvec(dY.data<float>(), dY.numel());
      EigenVectorArrayMap<float> dXvec(
          dX->template mutable_data<float>(), dX->numel());

      const float la = lambda_ * alpha_;
      dXvec = (Yvec > 0).select(lambda_ * dYvec, dYvec * (Yvec + la));
      return true;
        */
    }
}
