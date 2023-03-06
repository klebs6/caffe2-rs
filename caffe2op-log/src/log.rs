crate::ix!();

/**
  | Calculates the natural log of the given
  | input tensor ($ln(x)$), element-wise.
  | This operation can be done in an in-place
  | fashion too, by providing the same input
  | and output blobs.
  | 
  | Github Link:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/log_op.cc
  |
  */
pub struct LogFunctor<Context> {
    phantom: PhantomData<Context>,
}

num_inputs!{Log, 1}

num_outputs!{Log, 1}

inputs!{Log, 
    0 => ("X", "*(type: Tensor`<float>`)* Input tensor.")
}

outputs!{Log, 
    0 => ("Y", "*(type: Tensor`<float>`)* Output tensor computed as the natural log of the input tensor computed, element-wise.")
}

identical_type_and_shape!{Log}

allow_inplace!{Log, vec![(0, 0)]}

inherit_onnx_schema!{Log}

impl<Context> LogFunctor<Context> {

    #[inline] pub fn invoke<T>(&mut self, 
        n: i32, 
        x: *const T, 
        y: *mut T, 
        context: *mut Context) -> bool 
    {
        todo!();
        /*
            math::Log(N, X, Y, context);
            return true;
        */
    }
}

register_cpu_operator!{
    Log,
    UnaryElementwiseOp<
    TensorTypes<f32>, 
    CPUContext, 
    LogFunctor<CPUContext>>
}
