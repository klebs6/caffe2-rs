crate::ix!();

/**
  | Performs element-wise square-root
  | ($\sqrt{x}$) of input tensor $X$.
  | 
  | Github Link:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/sqrt_op.cc
  |
  */
pub struct SqrtFunctor<Context> { 

    /**
      | Input: X,
      | 
      | output: Y
      |
      */
    phantom: PhantomData<Context>,
}

num_inputs!{Sqrt, 1}

num_outputs!{Sqrt, 1}

inputs!{Sqrt, 
    0 => ("X", "*(type: Tensor`<float>`)* Input data tensor.")
}

outputs!{Sqrt, 
    0 => ("Y", "*(type: Tensor`<float>`)* Output tensor.")
}

identical_type_and_shape!{Sqrt}

allow_inplace!{Sqrt, vec![(0, 0)]}

register_cpu_operator!{Sqrt,
    UnaryElementwiseOp<
        TensorTypes<f32, f64>,
        CPUContext,
        SqrtFunctor<CPUContext>>
}

impl<Context> SqrtFunctor<Context> {

    #[inline] pub fn invoke<T>(&self, 
        n:       i32,
        x:       *const T,
        y:       *mut T,
        context: *mut Context) -> bool {

        todo!();
        /*
            math::Sqrt<T, Context>(N, X, Y, context);
        return true;
        */
    }
}
