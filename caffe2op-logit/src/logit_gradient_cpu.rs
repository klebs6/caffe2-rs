crate::ix!();

impl LogitGradientOp<f32, CPUContext> {

    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            const auto& X = Input(0);
      const auto& dY = Input(1);

      auto* dX = Output(0, X.sizes(), at::dtype<float>());
      int channels = X.dim32(X.dim() - 1);
      ConstEigenArrayMap<float> Xmat(
          X.template data<float>(), channels, X.numel() / channels);
      ConstEigenArrayMap<float> dYmat(
          dY.template data<float>(), channels, X.numel() / channels);
      EigenArrayMap<float> dXmat(
          dX->template mutable_data<float>(), channels, X.numel() / channels);
      dXmat = (Xmat < eps_ || Xmat > 1.0 - eps_)
                  .select(0, dYmat * ((1 - Xmat) * Xmat).inverse());
      return true;
        */
    }
}

register_cpu_operator!{
    Logit,
    UnaryElementwiseWithArgsOp<
        TensorTypes<f32>,
        CPUContext,
        LogitFunctor<CPUContext>>
}

register_cpu_operator!{
    LogitGradient, 
    LogitGradientOp<f32, CPUContext>
}
