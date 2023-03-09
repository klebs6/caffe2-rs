crate::ix!();

/**
  | Calculates the hyperbolic tangent
  | of the given input tensor element-wise.
  | 
  | This operation can be done in an in-place
  | fashion too, by providing the same input
  | and output blobs.
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/tanh_op.cc
  |
  */
pub struct TanhFunctor<Context> { 
    phantom: PhantomData<Context>,
}

num_inputs!{Tanh, 1}

num_outputs!{Tanh, 1}

inputs!{Tanh, 
    0 => ("input", "1-D input tensor")
}

outputs!{Tanh, 
    0 => ("output", "The hyperbolic tangent values of the input tensor, computed element-wise")
}

identical_type_and_shape!{Tanh}

allow_inplace!{Tanh, vec![(0, 0)]}

inherit_onnx_schema!{Tanh}

register_cpu_operator!{
    Tanh,
    UnaryElementwiseOp<
        TensorTypes<f32>,
        CPUContext,
        TanhFunctor<CPUContext>>
}

impl<Context> TanhFunctor<Context> {
    
    #[inline] pub fn invoke<T>(&self, 
        n:       i32,
        x:       *const T,
        y:       *mut T,
        context: *mut Context) -> bool {

        todo!();
        /*
            math::Tanh<T, Context>(N, X, Y, context);
        return true;
        */
    }
}
