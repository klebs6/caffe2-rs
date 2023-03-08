crate::ix!();

/**
  | ReluGradient takes both Y and dY and
  | uses this to update dX according to the
  | chain rule and derivatives of the rectified
  | linear function.
  |
  */
pub struct ReluNGradientFunctor<Context> {
    n:  f32,

    /**
      | Input: Y, dY
      | 
      | output: dX
      |
      */
    phantom: PhantomData<Context>,
}

num_inputs!{ReluNGradient, 2}

num_outputs!{ReluNGradient, 1}

args!{ReluNGradient, 
    0 => ("n", "the cap of forward op output")
}

allow_inplace!{ReluNGradient, vec![(1, 0)]}

impl<Context> ReluNGradientFunctor<Context> {

    pub fn new(op: &mut OperatorStorage) -> Self {
    
        todo!();
        /*
            : n(op.GetSingleArgument<f32>("n", 6.0f)) 

        CAFFE_ENFORCE_GT(n, 0, "n should be greater than 0");
        */
    }
}

impl ReluNGradientFunctor<CPUContext> {

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
      ConstEigenVectorArrayMap<T> Y_arr(Y, size);
      EigenVectorArrayMap<T>(dX, size) =
          (Y_arr > T(0) && Y_arr < T(n))
              .select(ConstEigenVectorArrayMap<T>(dY, size), T(0));
      return true;
        */
    }
}
