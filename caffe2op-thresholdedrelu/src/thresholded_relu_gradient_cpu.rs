crate::ix!();

impl ThresholdedReluGradientOp<f32,CPUContext> {

    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& Y = Input(0);
      auto& dY = Input(1);

      CAFFE_ENFORCE_EQ(dY.numel(), Y.numel());
      auto* dX = Output(0, Y.sizes(), at::dtype<float>());

      const float* Ydata = Y.data<float>();
      const float* dYdata = dY.data<float>();
      float* dXdata = dX->template mutable_data<float>();
      EigenVectorArrayMap<float> dXvec(dXdata, dX->numel());
      ConstEigenVectorArrayMap<float> Yvec(Ydata, Y.numel());
      ConstEigenVectorArrayMap<float> dYvec(dYdata, dY.numel());
      dXvec = dYvec * Yvec.cwiseSign();
      /* Non vectorized implementation
      for (int i = 0; i < Y.size(); ++i) {
        dXdata[i] = Ydata[i] > 0 ? dYdata[i] : 0;
      }
      */
      return true;
        */
    }
}

register_cpu_operator!{
    ThresholdedReluGradient,
    ThresholdedReluGradientOp<f32, CPUContext>
}
