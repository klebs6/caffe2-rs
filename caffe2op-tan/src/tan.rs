crate::ix!();

/**
  | Calculates the tangent of the given
  | input tensor, element-wise.
  |
  */
pub struct TanFunctor<Context> { 

    phantom: PhantomData<Context>,
}

register_cpu_operator!{
    Tan,
    UnaryElementwiseOp<TensorTypes<f32>, CPUContext, TanFunctor<CPUContext>>
}

num_inputs!{Tan, 1}

num_outputs!{Tan, 1}

inputs!{Tan, 
    0 => ("input", "Input tensor")
}

outputs!{Tan, 
    0 => ("output", "The tangent of the input tensor computed element-wise")
}

identical_type_and_shape!{Tan}

impl<Context> TanFunctor<Context> {

    pub fn call<T>(
        n:       i32,
        x:       *const T,
        y:       *mut T,
        context: *mut Context) -> bool 
    {
        todo!();
        /*
           math::Tan(N, X, Y, context);
           return true;
           */
    }
}
