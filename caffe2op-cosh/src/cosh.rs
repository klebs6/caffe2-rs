crate::ix!();

/**
  | Calculates the hyperbolic cosine of
  | the given input tensor, element-wise.
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/cosh_op.cc
  |
  */
pub struct CoshFunctor<Context> {
    phantom: PhantomData<Context>,
}

num_inputs!{Cosh, 1}

num_outputs!{Cosh, 1}

inputs!{Cosh, 
    0 => ("input", "Input tensor")
}

outputs!{Cosh, 
    0 => ("output", "The hyperbolic cosine values of the input tensor, computed element-wise")
}

identical_type_and_shape!{Cosh}

inherit_onnx_schema!{Cosh}

impl<Context> CoshFunctor<Context> {

    #[inline] pub fn invoke<T>(
        &mut self, 
        n:         i32, 
        x:         *const T, 
        y:         *mut T, 
        context:   *mut Context) -> bool 
    {
        todo!();
        /*
            math::Cosh(N, X, Y, context);
            return true;
        */
    }
}
