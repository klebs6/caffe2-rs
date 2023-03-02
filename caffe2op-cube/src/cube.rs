crate::ix!();

pub struct CubeFunctor<Context> {
    phantom: PhantomData<Context>,
}

impl<Context> CubeFunctor<Context> {

    #[inline] pub fn invoke<T>(
        &self,
        n:       i32,
        x:       *const T,
        y:       *mut T,
        context: *mut Context) -> bool 
    {
        todo!();
        /*
            math::Cube<T, Context>(N, X, Y, context);
            return true;
        */
    }
}

register_cpu_operator!{
    Cube,
    UnaryElementwiseOp<NumericTypes, CPUContext, CubeFunctor<CPUContext>>
}

num_inputs!{Cube, 1}

num_outputs!{Cube, 1}

inputs!{Cube, 
    0 => ("X", "*(type: Tensor`<float>`)* Input tensor.")
}

outputs!{Cube, 
    0 => ("Y", "*(type: Tensor`<float>`)* Output tensor calculated as the cube of the input tensor, element-wise.")
}

identical_type_and_shape!{Cube}
