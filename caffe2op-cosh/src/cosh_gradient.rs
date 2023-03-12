crate::ix!();

pub struct CoshGradientFunctor<Context> {
    phantom: PhantomData<Context>,
}

num_inputs!{CoshGradient, 2}

num_outputs!{CoshGradient, 1}

identical_type_and_shape_of_input!{CoshGradient, 0}

impl CoshGradientFunctor<CPUContext> {

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
            const int size = std::accumulate(
              X_dims.cbegin(), X_dims.cend(), 1, std::multiplies<int>());
          ConstEigenVectorArrayMap<T> dY_arr(dY, size);
          ConstEigenVectorArrayMap<T> X_arr(X, size);
          EigenVectorMap<T>(dX, size) = dY_arr * (X_arr.exp() - (-X_arr).exp()) / 2;
          return true;
        */
    }
}
