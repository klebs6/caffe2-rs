crate::ix!();

impl HardSigmoidGradientFunctor<CPUContext> {

    #[inline] pub fn forward<T>(&self, 
        y_dims:   &Vec<i32>,
        dY_dims:  &Vec<i32>,
        y:        *const T,
        dY:       *const T,
        dX:       *mut T,
        context:  *mut CPUContext) -> bool 
    {
        todo!();
        /*
            const int size = std::accumulate(
              Y_dims.cbegin(), Y_dims.cend(), 1, std::multiplies<int>());
          ConstEigenVectorArrayMap<T> Y_arr(Y, size);
          EigenVectorArrayMap<T>(dX, size) =
              (Y_arr > T(0) && Y_arr < T(1))
                  .select(ConstEigenVectorArrayMap<T>(dY, size) * alpha, T(0));
          return true;
        */
    }
}
