crate::ix!();

/**
  | Calculates the absolute value of the
  | given input tensor, element-wise.
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/abs_op.cc
  |
  */
pub struct AbsFunctor<Context> {
    context: Context,
}

num_inputs!{Abs, 1}

num_outputs!{Abs, 1}

inputs!{Abs, 
    0 => ("X", "*(type: Tensor<float>)* Input tensor.")
}

outputs!{Abs, 
    0 => ("Y", "*(type: Tensor`<float>`)* Absolute value of input element-wise.")
}

identical_type_and_shape!{Abs}

inherit_onnx_schema!{Abs}

impl<Context> AbsFunctor<Context> {

    #[inline] pub fn invoke<T>(
        &self, 
        n: i32,
        x: *const T,
        y: *mut T,
        context: *mut Context) -> bool 
    {
        todo!();
        /*
            math::Abs(N, X, Y, context);
            return true;
        */
    }
}
