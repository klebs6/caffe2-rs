crate::ix!();

pub struct ReciprocalGradientFunctor<Context> {
    phantom: PhantomData<Context>,
}

impl<CPUContext> ReciprocalGradientFunctor<CPUContext> {

    #[inline] pub fn forward<T>(
        y_dims:  &Vec<i32>,
        dY_dims: &Vec<i32>,
        y:       *const T,
        dY:      *const T,
        dX:      *mut T,
        context: *mut CPUContext) -> bool {

        todo!();
        /*
          const int size = std::accumulate(
              Y_dims.cbegin(), Y_dims.cend(), 1, std::multiplies<int>());
          ConstEigenVectorArrayMap<T> dY_arr(dY, size);
          ConstEigenVectorArrayMap<T> Y_arr(Y, size);
          EigenVectorMap<T>(dX, size) = dY_arr * (-Y_arr.square());
          return true;
        */
    }
}
