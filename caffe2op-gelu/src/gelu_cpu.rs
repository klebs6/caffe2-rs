crate::ix!();

impl GeluFunctor<CPUContext> {

    #[inline] pub fn invoke<T>(
        &mut self,
        n:        i32,
        x:        *const T,
        y:        *mut T,
        context:  *mut CPUContext) -> bool 
    {
        todo!();
        /*
            if (fast_gelu) {
            // y = 0.5x * (1 + tanh(sqrt(2/Pi) * (x + 0.044715x^3)))
            constexpr T kAlpha = M_2_SQRTPI * M_SQRT1_2;
            ConstEigenVectorArrayMap<T> X_arr(X, N);
            EigenVectorArrayMap<T> Y_arr(Y, N);
            Y_arr = X_arr *
                (((X_arr + X_arr.cube() * gelu_utils::kFastCoeff) * kAlpha).tanh() +
                 T(1)) *
                static_cast<T>(0.5);
          } else {
            // y = x * P(X <= x) where X ~ N(0, 1)
            math::CdfNorm<T, CPUContext>(N, X, Y, context);
            math::Mul<T, CPUContext>(N, X, Y, Y, context);
          }
          return true;
        */
    }
}
