crate::ix!();

/**
  | Calculates the arccosine of the given
  | input tensor, element-wise.
  |
  */
pub struct AcosFunctor<Context> {
    phantom: PhantomData<Context>,

}

num_inputs!{Acos, 1}

num_outputs!{Acos, 1}

inputs!{Acos, 
    0 => ("input", "Input tensor")
}

outputs!{Acos, 
    0 => ("output", "The arccosine of the input tensor computed element-wise")
}

identical_type_and_shape!{Acos}

register_cpu_operator!{
    Acos,
    UnaryElementwiseOp<
        TensorTypes<f32>,
        CPUContext,
        AcosFunctor<CPUContext>>
}

impl<Context> AcosFunctor<Context> {

    #[inline] pub fn invoke<T>(
        n:       i32,
        x:       *const T,
        y:       *mut T,
        context: *mut Context) -> bool 
    {
        todo!();
        /*
            math::Acos(N, X, Y, context);
            return true;
        */
    }
}

