crate::ix!();

impl<CPUContext> EluGradientFunctor<CPUContext> {
    #[inline] pub fn forward<T>(
        &self,
        y_dims:    &Vec<i32>,
        dY_dims:   &Vec<i32>,
        y:         *const T,
        dY:        *const T,
        dX:        *mut T,
        context:   *mut CPUContext) -> bool 
    {
        todo!();
        /*
            const int size = std::accumulate(
              Y_dims.cbegin(), Y_dims.cend(), 1, std::multiplies<int>());
          ConstEigenVectorArrayMap<T> Y_arr(Y, size);
          ConstEigenVectorArrayMap<T> dY_arr(dY, size);
          EigenVectorArrayMap<T>(dX, size) =
              (Y_arr < 0).select(dY_arr * (Y_arr + alpha), dY_arr);
          return true;
        */
    }
}
