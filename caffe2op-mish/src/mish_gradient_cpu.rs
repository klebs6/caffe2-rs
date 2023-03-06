crate::ix!();

impl MishGradientOp<CPUContext> {

    #[inline] pub fn do_run_with_type<T>(&mut self) -> bool {
        todo!();
        /*
            const auto& X = Input(INPUT);
      const auto& Y = Input(OUTPUT);
      const auto& dY = Input(OUTPUT_GRAD);

      CAFFE_ENFORCE_EQ(X.numel(), Y.numel());
      CAFFE_ENFORCE_EQ(dY.numel(), Y.numel());
      auto* dX = Output(INPUT_GRAD, X.sizes(), at::dtype<T>());

      const T* X_data = X.template data<T>();
      const T* Y_data = Y.template data<T>();
      const T* dY_data = dY.template data<T>();
      T* dX_data = dX->template mutable_data<T>();

      const int64_t N = X.numel();
      ConstEigenVectorArrayMap<T> X_arr(X_data, N);
      ConstEigenVectorArrayMap<T> Y_arr(Y_data, N);
      ConstEigenVectorArrayMap<T> dY_arr(dY_data, N);
      EigenVectorArrayMap<T> dX_arr(dX_data, N);

      math::Exp<T, CPUContext>(N, X_data, dX_data, &context_);
      math::Log1p<T, CPUContext>(N, dX_data, dX_data, &context_);
      math::Tanh<T, CPUContext>(N, dX_data, dX_data, &context_);
      dX_arr = dY_arr *
          (dX_arr +
           X_arr * (T(1) - dX_arr.square()) * T(0.5) *
               ((X_arr * T(0.5)).tanh() + T(1)));

      return true;
        */
    }
}
