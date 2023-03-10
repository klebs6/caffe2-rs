crate::ix!();

/**
  | Calculates the arcsine of the given
  | input tensor, element-wise.
  |
  */
pub struct AsinFunctor<Context> {
    phantom: PhantomData<Context>,
}

num_inputs!{Asin, 1}

num_outputs!{Asin, 1}

inputs!{Asin, 
    0 => ("input", "Input tensor")
}

outputs!{Asin, 
    0 => ("output", "The arcsine of the input tensor computed element-wise")
}

identical_type_and_shape!{Asin}

register_cpu_operator!{
    Asin,
    UnaryElementwiseOp<
        TensorTypes<f32>,
        CPUContext,
        AsinFunctor<CPUContext>>
}

impl<Context> AsinFunctor<Context> {

    #[inline] pub fn invoke<T>(
        &mut self,
        n:       i32,
        x:       *const T,
        y:       *mut T,
        context: *mut Context) -> bool 
    {
        todo!();
        /*
            math::Asin(N, X, Y, context);
            return true;
        */
    }
}
