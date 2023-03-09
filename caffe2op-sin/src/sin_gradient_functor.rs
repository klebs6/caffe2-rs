crate::ix!();

pub struct SinGradientFunctor<Context> {
    phantom: PhantomData<Context>,
}

impl SinGradientFunctor<CPUContext> {
    
    #[inline] pub fn forward<T>(&self, 
        x_dims:  &Vec<i32>,
        dy_dims: &Vec<i32>,
        x:       *const T,
        dy:      *const T,
        dx:      *mut T,
        context: *mut CPUContext) -> bool {

        todo!();
        /*
          const int size = std::accumulate(
          X_dims.cbegin(), X_dims.cend(), 1, std::multiplies<int>());
          ConstEigenVectorArrayMap<T> dY_arr(dY, size);
          ConstEigenVectorArrayMap<T> X_arr(X, size);
          EigenVectorMap<T>(dX, size) = dY_arr * X_arr.cos();
          return true;
        */
    }
}
