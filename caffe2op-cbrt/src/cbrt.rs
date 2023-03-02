crate::ix!();

pub struct CbrtFunctor<Context> {
    phantom: PhantomData<Context>,
}

register_cpu_operator!{
    Cbrt,
    UnaryElementwiseOp<
        TensorTypes<f32>,
        CPUContext,
        CbrtFunctor<CPUContext>>
}

num_inputs!{Cbrt, 1}

num_outputs!{Cbrt, 1}

inputs!{Cbrt, 
    0 => ("X", "*(type: Tensor`<float>`)* Input tensor.")
} 

outputs!{Cbrt, 
    0 => ("Y", "*(type: Tensor`<float>`)* Output tensor calculated as the cbrt of the input tensor, element-wise.")
} 

allow_inplace!{Cbrt, vec![(0, 0)]}

identical_type_and_shape!{Cbrt} 

impl<Context> CbrtFunctor<Context> {

    #[inline] pub fn invoke<T>(
        &self, 
        n: i32,
        x: *const T,
        y: *mut T,
        context: *mut Context) -> bool 
    {
        todo!();
        /*
            math::Cbrt<T, Context>(N, X, Y, context);
            return true;
        */
    }
}
