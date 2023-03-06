crate::ix!();

impl GeluGradientFunctor<CPUContext> {

    #[inline] pub fn forward<T>(
        &self,
        dY_dims:   &Vec<i32>,
        x_dims:    &Vec<i32>,
        dY:        *const T,
        x:         *const T,
        dX:        *mut T,
        context:   *mut CPUContext) -> bool 
    {
        todo!();
        /*
            const int N = std::accumulate(
              dY_dims.cbegin(), dY_dims.cend(), 1, std::multiplies<int>());
          ConstEigenVectorArrayMap<T> dY_arr(dY, N);
          ConstEigenVectorArrayMap<T> X_arr(X, N);
          EigenVectorArrayMap<T> dX_arr(dX, N);
          if (fast_gelu) {
            constexpr T kAlpha = M_2_SQRTPI * M_SQRT1_2;
            constexpr T kBeta = kAlpha * gelu_utils::kFastCoeff * T(3);
            dX_arr = ((X_arr + X_arr.cube() * gelu_utils::kFastCoeff) * kAlpha).tanh();
            dX_arr =
                (T(1) + dX_arr +
                 X_arr * (T(1) - dX_arr.square()) * (kBeta * X_arr.square() + kAlpha)) *
                dY_arr * static_cast<T>(0.5);
          } else {
            constexpr T kAlpha = M_2_SQRTPI * M_SQRT1_2 * T(0.5);
            math::CdfNorm<T, CPUContext>(N, X, dX, context);
            dX_arr = (dX_arr +
                      X_arr * (-X_arr.square() * static_cast<T>(0.5)).exp() * kAlpha) *
                dY_arr;
          }
          return true;
        */
    }
}
