crate::ix!();

register_cpu_operator!{
    RMSNorm,
    RMSNormOp<CPUContext>
}

register_cpu_operator!{
    RMSNormGradient, 
    RMSNormGradientOp<CPUContext>
}

impl RMSNormOp<CPUContext> {

    #[inline] pub fn do_run_with_type<T>(&mut self) -> bool {
        todo!();
        /*
            const auto& X = Input(0);
          const auto& gamma = Input(1);
          const auto& beta = Input(2);
          auto* Y = Output(0, X.sizes(), at::dtype<T>());
          CAFFE_ENFORCE_GE(X.dim(), 2, "RMSNorm requires input dim >= 2.");
          const int canonical_axis = X.canonical_axis_index(axis_);
          const std::vector<int64_t> rms_dims(
              X.sizes().cbegin(), X.sizes().cbegin() + canonical_axis);
          auto* rrms = Output(1, rms_dims, at::dtype<T>());
          const int64_t M = X.size_to_dim(canonical_axis);
          const int64_t N = X.size_from_dim(canonical_axis);
          CAFFE_ENFORCE_EQ(gamma.numel(), N);
          CAFFE_ENFORCE_EQ(beta.numel(), N);

          const T* X_data = X.template data<T>();
          const T* gamma_data = gamma.template data<T>();
          const T* beta_data = beta.template data<T>();
          T* Y_data = Y->template data<T>();
          T* rrms_data = rrms->template data<T>();

          ConstEigenArrayMap<T> X_arr(X_data, N, M);
          ConstEigenVectorArrayMap<T> gamma_arr(gamma_data, N);
          ConstEigenVectorArrayMap<T> beta_arr(beta_data, N);
          EigenArrayMap<T> Y_arr(Y_data, N, M);
          at::parallel_for(0, M, 1, [&](int64_t start, int64_t end) {
            for (int64_t i = start; i < end; ++i) {
              const T rrms_val =
                  T(1) / std::sqrt(X_arr.col(i).square().mean() + static_cast<T>(eps_));
              Y_arr.col(i) = rrms_val * X_arr.col(i) * gamma_arr + beta_arr;
              rrms_data[i] = rrms_val;
            }
          });

          return true;
        */
    }
}
