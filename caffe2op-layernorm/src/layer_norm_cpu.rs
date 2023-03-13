crate::ix!();

register_cpu_operator!{
    LayerNorm, 
    LayerNormOp<CPUContext>
}

impl LayerNormOp<CPUContext> {

    #[inline] pub fn compute_sigma_and_fused_params<T>(
        &mut self,
        n:       i32,
        eps:     f32,
        mean:    *const T,
        var:     *const T,
        sigma:   *mut T,
        scale:   *mut T,
        bias:    *mut T) 
    {
        todo!();
        /*
            ConstEigenVectorArrayMap<T> var_arr(var, N);
          EigenVectorArrayMap<T> sigma_arr(sigma, N);
          sigma_arr = var_arr + static_cast<T>(eps);
          math::Rsqrt<T, CPUContext>(N, sigma, scale, &context_);
          math::Mul<T, CPUContext>(N, scale, sigma, sigma, &context_);
          EigenVectorArrayMap<T>(bias, N) = -ConstEigenVectorArrayMap<T>(scale, N) *
              ConstEigenVectorArrayMap<T>(mean, N);
        */
    }

    #[inline] pub fn layer_norm_forward<T>(
        &mut self, 
        m:      i32,
        n:      i32,
        x:      *const T,
        scale:  *const T,
        bias:   *const T,
        gamma:  *const T,
        beta:   *const T,
        y:      *mut T) 
    {
        todo!();
        /*
            ConstEigenArrayMap<T> X_arr(X, N, M);
          ConstEigenVectorArrayMap<T> scale_arr(scale, M);
          ConstEigenVectorArrayMap<T> bias_arr(bias, M);
          EigenArrayMap<T> Y_arr(Y, N, M);
          if (gamma != nullptr && beta != nullptr) {
            ConstEigenVectorArrayMap<T> gamma_arr(gamma, N);
            ConstEigenVectorArrayMap<T> beta_arr(beta, N);
            Y_arr = (((X_arr.rowwise() * scale_arr.transpose()).rowwise() +
                      bias_arr.transpose())
                         .colwise() *
                     gamma_arr)
                        .colwise() +
                beta_arr;
          } else {
            CAFFE_ENFORCE(gamma == nullptr);
            CAFFE_ENFORCE(beta == nullptr);
            Y_arr = (X_arr.rowwise() * scale_arr.transpose()).rowwise() +
                bias_arr.transpose();
          }
        */
    }
}

