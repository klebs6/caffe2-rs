crate::ix!();

/**
  | ReluGradient takes both Y and dY and
  | uses this to update dX according to the
  | chain rule and derivatives of the rectified
  | linear function.
  |
  */
pub struct ReluGradientFunctor<Context> {

    /**
      | Input: Y, dY
      | 
      | output: dX
      |
      */
    phantom: PhantomData<Context>,
}

num_inputs!{ReluGradient, 2}

num_outputs!{ReluGradient, 1}

identical_type_and_shape_of_input!{ReluGradient, 1}

allow_inplace!{ReluGradient, vec![(1, 0)]}

impl ReluGradientFunctor<CPUContext> {

    #[inline] pub fn forward<T>(&self, 
        y_dims:  &Vec<i32>,
        dy_dims: &Vec<i32>,
        y:       *const T,
        dy:      *const T,
        dx:      *mut T,
        context: *mut CPUContext) -> bool {
    
        todo!();
        /*
            const int size = std::accumulate(
          Y_dims.cbegin(), Y_dims.cend(), 1, std::multiplies<int>());
      EigenVectorArrayMap<T>(dX, size) =
          (ConstEigenVectorArrayMap<T>(Y, size) > T(0))
              .select(ConstEigenVectorArrayMap<T>(dY, size), T(0));
      return true;
        */
    }
}
