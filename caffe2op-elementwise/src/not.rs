crate::ix!();

/**
  | Performs element-wise negation on
  | input tensor `X`.
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/elementwise_ops_schema.cc
  |
  */
pub struct NotFunctor<Context> {
    phantom: PhantomData<Context>,
}

num_inputs!{Not, 1}

num_outputs!{Not, 1}

inputs!{Not, 
    0 => ("X", "*(Tensor`<bool>`)* Input tensor.")
}

outputs!{Not, 
    0 => ("Y", "*(Tensor`<bool>`)* Negated output tensor.")
}

identical_type_and_shape_of_input!{Not, 0}

inherit_onnx_schema!{Not}

should_not_do_gradient!{Not}

impl<Context> NotFunctor<Context> {
    
    #[inline] pub fn invoke(
        &self, 
        n:       i32,
        x:       *const bool,
        y:       *mut bool,
        context: *mut Context) -> bool 
    {
        todo!();
        /*
            math::Not(N, X, Y, context);
        return true;
        */
    }
}

