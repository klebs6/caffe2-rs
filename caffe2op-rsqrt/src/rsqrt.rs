crate::ix!();

/**
  | Computes the element-wise rsqrt of
  | the input.
  |
  */
pub struct RsqrtFunctor<Context> {
    
    phantom: PhantomData<Context>,
}

num_inputs!{Rsqrt, 1}

num_outputs!{Rsqrt, 1}

inputs!{Rsqrt, 
    0 => ("X", "ND input tensor")
}

outputs!{Rsqrt, 
    0 => ("Y", "ND output tensor")
}

identical_type_and_shape!{Rsqrt}

allow_inplace!{Rsqrt, vec![(0, 0)]}

impl<Context> RsqrtFunctor<Context> {

    #[inline] pub fn invoke<T>(&self, 
        n:       i32,
        x:       *const T,
        y:       *mut T,
        context: *mut Context) -> bool {

        todo!();
        /*
            math::Rsqrt<T, Context>(N, X, Y, context);
        return true;
        */
    }
}
