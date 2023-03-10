crate::ix!();

/**
  | Calculates the arctangent of the given
  | input tensor, element-wise.
  |
  */
pub struct AtanFunctor<Context> {
    phantom: PhantomData<Context>,
}

num_inputs!{Atan, 1}

num_outputs!{Atan, 1}

inputs!{Atan, 
    0 => ("input", "Input tensor")
}

outputs!{Atan, 
    0 => ("output", "The arctangent of the input tensor computed element-wise")
}

identical_type_and_shape!{Atan}

register_cpu_operator!{
    Atan,
    UnaryElementwiseOp<
        TensorTypes<f32>,
        CPUContext,
        AtanFunctor<CPUContext>>
}

impl<Context> AtanFunctor<Context> {

    #[inline] pub fn invoke<T>(
        &mut self,
        n:       i32,
        x:       *const T,
        y:       *mut T,
        context: *mut Context) -> bool 
    {
        todo!();
        /*
            math::Atan(N, X, Y, context);
            return true;
        */
    }
}
