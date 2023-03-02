crate::ix!();

/**
  | Calculates the exponential of the given
  | input tensor ($exp(x)$), element-wise.
  | This operation can be done in an in-place
  | fashion too, by providing the same input
  | and output blobs.
  | 
  | Github Link:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/exp_op.cc
  |
  */
pub struct ExpFunctor<Context> {
    phantom: PhantomData<Context>,
}

impl<Context> ExpFunctor<Context> {

    #[inline] pub fn invoke<T>(
        &self,
        n:       i32,
        x:       *const T,
        y:       *mut T,
        context: *mut Context) -> bool 
    {
        todo!();
        /*
            math::Exp(N, X, Y, context);
            return true;
        */
    }
}

register_cpu_operator!{
    Exp,
    UnaryElementwiseOp<TensorTypes<f32>, CPUContext, ExpFunctor<CPUContext>>
}

num_inputs!{Exp, 1}

num_outputs!{Exp, 1}

inputs!{Exp, 
    0 => ("X", "*(type: Tensor`<float>`)* Input tensor.")
}

outputs!{Exp, 
    0 => ("Y", "*(type: Tensor`<float>`)* The exponential of the input tensor computed element-wise.")
}

identical_type_and_shape!{Exp}

inherit_onnx_schema!{Exp}

allow_inplace!{Exp, vec![(0, 0)]}
