crate::ix!();

/**
  | Calculates the cosine of the given input
  | tensor, element-wise.
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/cos_op.cc
  |
  */
pub struct CosFunctor<Context> {
    phantom: PhantomData<Context>,
}

num_inputs!{Cos,  1}

num_outputs!{Cos, 1}

inputs!{Cos, 
    0 => ("X", "*(type: Tensor`<float>`)* Input tensor.")
}

outputs!{Cos, 
    0 => ("Y", "*(type: Tensor`<float>`)* Output tensor calculated as the cosine of the input tensor, element-wise.")
}

identical_type_and_shape!{Cos}

register_cpu_operator!{
    Cos,
    UnaryElementwiseOp<
        TensorTypes<f32>, 
        CPUContext, 
        CosFunctor<CPUContext>
    >
}

impl<Context> CosFunctor<Context> {

    #[inline] pub fn invoke<T>(
        n:       i32,
        x:       *const T,
        y:       *mut T,
        context: *mut Context) -> bool 
    {
        todo!();
        /*
            math::Cos(N, X, Y, context);
            return true;
        */
    }
}
